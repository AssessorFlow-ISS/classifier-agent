"""Golden Workflow Evaluation — single workflow, all agents.

Triggers a real assessment workflow in af-golden, waits for completion,
fetches Langfuse traces, evaluates each agent's output with GEval metrics,
and writes per-agent baselines.

Usage (inside K8s Job):
  python golden_workflow_eval.py

Env vars:
  API_SERVER_URL        — golden API server (http://api-server-golden:8001)
  ORCHESTRATOR_DB_*     — golden Orchestrator DB connection
  LANGFUSE_HOST/KEY     — Langfuse for trace retrieval
  BASELINE_DIR          — where to write baseline JSON files
  WORKFLOW_TIMEOUT_S    — max seconds to wait for workflow (default 300)
  GOLDEN_SCENARIO       — "insufficient" or "sufficient" (default insufficient)
  DEEPEVAL_JUDGE_MODEL  — model for GEval judge (e.g. gemini-2.5-pro)
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# ── Configuration ──
#
# Lessons-learned from CI fix (drift_runner.py / deepteam_smoke.py):
# every secret + service URL is REQUIRED via os.environ[] — silent defaults
# mask misconfiguration and let CI silently produce wrong results.
# Run-config that's safe to default (BASELINE_DIR, TIMEOUT, SCENARIO,
# GOLDEN_PARTICIPANT_EMAIL) keeps its default — those don't affect what
# the test actually exercises.

# REQUIRED — secrets + service URLs (raise on missing)
API_SERVER = os.environ["API_SERVER_URL"]
LANGFUSE_HOST = os.environ["LANGFUSE_HOST"]
LANGFUSE_PK = os.environ["LANGFUSE_PUBLIC_KEY"]
LANGFUSE_SK = os.environ["LANGFUSE_SECRET_KEY"]

# OPTIONAL — run-config defaults (safe to fall back; not security-critical)
BASELINE_DIR = Path(os.environ.get("BASELINE_DIR", "/tmp/baselines"))
TIMEOUT = int(os.environ.get("WORKFLOW_TIMEOUT_S", "300"))
SCENARIO = os.environ.get("GOLDEN_SCENARIO", "insufficient")
JUDGE_MODEL = os.environ.get("DEEPEVAL_JUDGE_MODEL")

# Insufficient scenario: small material, 15 questions, hard difficulty
# This exercises: Validator (PROCEED), Classification (INSUFFICIENT + autonomy),
# Q&A Gen (won't run if insufficient), Evaluator (won't run), Reporting (won't run)
# But for sufficient scenario: exercises ALL agents through the full pipeline.
GOLDEN_PARTICIPANT = os.environ.get("GOLDEN_PARTICIPANT_EMAIL", "golden@baseline.test")

SCENARIOS = {
    "insufficient": {
        "assessment_title": f"Golden Baseline (Insufficient) {datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}",
        "purpose": "skill_assessment",
        "duration_minutes": 30,
        "difficulty_level": "hard",
        "structured_question_count": 16,
        "non_structured_question_count": 2,
        "participants": [GOLDEN_PARTICIPANT],
        "web_research_mode": "auto",
    },
    "sufficient": {
        "assessment_title": f"Golden Baseline (Sufficient) {datetime.now(timezone.utc).strftime('%Y%m%d-%H%M')}",
        "purpose": "skill_assessment",
        "duration_minutes": 30,
        "difficulty_level": "medium",
        "structured_question_count": 8,
        "non_structured_question_count": 2,
        "participants": [GOLDEN_PARTICIPANT],
        "web_research_mode": "manual",
    },
}

# Per-agent metrics to evaluate from Langfuse traces
# Each agent has domain-specific GEval criteria (defined in _evaluate_agent)
# "validator-moderation" is a separate eval target for the OpenAI Moderation API
# calls within the Validator workflow (judged by Gemini)
AGENT_METRICS = {
    "validator": ["safety_reasoning"],
    "validator-moderation": ["moderation_accuracy"],
    "classification": ["sufficiency_quality"],
    "qna-generation": ["question_relevance"],
    "evaluator": ["grading_quality"],
    "web-research": ["research_relevance"],
    "reporting": ["report_quality"],
}

# Agent name patterns for filtering Langfuse observations
AGENT_PATTERNS = {
    "validator": ["validator", "content_safety", "mrc", "ocr", "content_analysis", "synthesizer"],
    "validator-moderation": ["moderation", "text_moderation", "image_moderation"],
    "classification": ["classification", "sufficiency", "topic_extraction", "rubric_fitness"],
    "qna-generation": ["qna", "q&a", "question", "generation"],
    "evaluator": ["evaluator", "grading", "quality_validation"],
    "web-research": ["web_research", "web-research", "search"],
    "reporting": ["reporting", "report", "topic_gap"],
}


def _api(method: str, path: str, body: dict | None = None) -> dict:
    """Call the golden API server.

    The api-server's UNAUTHENTICATED guard requires either an
    X-Forwarded-Email header (set by oauth2-proxy in production) or an
    assessor_id field in the body. The runner port-forwards directly to
    the pod, bypassing oauth2-proxy, so we pass X-Forwarded-Email
    explicitly. Override via GOLDEN_ASSESSOR_EMAIL env if needed.
    """
    url = f"{API_SERVER}{path}"
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header(
        "X-Forwarded-Email",
        os.environ.get("GOLDEN_ASSESSOR_EMAIL", "golden-assessor@baseline.test"),
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def _langfuse_get(path: str) -> dict:
    """Call Langfuse API."""
    auth = base64.b64encode(f"{LANGFUSE_PK}:{LANGFUSE_SK}".encode()).decode()
    url = f"{LANGFUSE_HOST}{path}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Basic {auth}")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


async def _get_phase(dsn: str, workflow_id: str) -> str:
    """Get current workflow phase from Orchestrator DB."""
    import asyncpg
    try:
        conn = await asyncpg.connect(dsn)
        try:
            row = await conn.fetchrow(
                "SELECT current_phase FROM workflows WHERE id = $1",
                workflow_id,
            )
            return row["current_phase"] if row else "unknown"
        finally:
            await conn.close()
    except Exception as e:
        print(f"  DB error: {e}")
        return "unknown"


async def _wait_for_phases(dsn: str, workflow_id: str, targets: list[str], timeout_s: int) -> str:
    """Poll until workflow reaches one of the target phases.

    Checks BOTH the DB phase column AND workflow events, because the
    Orchestrator may advance the workflow (emit events, send invitations)
    without updating the DB phase column.
    """
    deadline = time.time() + timeout_s
    phase = "unknown"
    last_printed = ""
    while time.time() < deadline:
        phase = await _get_phase(dsn, workflow_id)
        if phase != last_printed:
            print(f"  Phase: {phase}")
            last_printed = phase
        if phase in targets:
            return phase

        # Check events for logical progress beyond the DB phase
        try:
            events = _api("GET", f"/api/v1/workflows/{workflow_id}/events")
            event_list = events if isinstance(events, list) else events.get("events", events.get("data", []))
            event_types = {e.get("event_type", "") for e in event_list}
            # If invitations sent, workflow is logically at assessment_active
            if "orchestrator.invitation_sent" in event_types and "assessment_active" in targets:
                print(f"  Phase: assessment_active (from events, DB shows {phase})")
                return "assessment_active"
            # If reports distributed, workflow is logically completed
            if "assessorflow.reports.distributed" in event_types and "completed" in targets:
                print(f"  Phase: completed (from events, DB shows {phase})")
                return "completed"
            # If report review pending, reporting is done and waiting for HITL Gate 2
            if "assessorflow.report-review.pending" in event_types and "report_review" in targets:
                print(f"  Phase: report_review (from events, DB shows {phase})")
                return "report_review"
            # If reporting complete event, reporting agent has finished
            if "assessorflow.reporting.complete" in event_types and "report_review" in targets:
                print(f"  Phase: report_review (from reporting.complete event, DB shows {phase})")
                return "report_review"
            # If HITL review event exists, workflow reached human_review
            if "assessorflow.hitl-review.pending" in event_types and "human_review" in targets:
                print(f"  Phase: human_review (from events, DB shows {phase})")
                return "human_review"
        except Exception:
            pass  # API unavailable, fall back to DB polling only

        await asyncio.sleep(5)
    print(f"  TIMEOUT after {timeout_s}s (last phase: {phase})")
    return phase


async def _run_full_pipeline(workflow_id: str, assessment_id: str) -> str:
    """Drive the workflow through ALL phases including HITL gates.

    Matches the T1-full 13-stage lifecycle in extended-flows.spec.ts:
      material_validation → classification → question_generation → quality_validation
      → human_review → APPROVE → distribute → participant submit → grading
      → reporting → report_review → DISTRIBUTE REPORTS → completed
    """
    import asyncpg

    # All DB connection params REQUIRED — `dev_password` default would mask
    # a CI-secrets misconfiguration and silently produce wrong results.
    host = os.environ["ORCHESTRATOR_DB_HOST"]
    port = os.environ["ORCHESTRATOR_DB_PORT"]
    name = os.environ["ORCHESTRATOR_DB_NAME"]
    user = os.environ["ORCHESTRATOR_DB_USER"]
    password = os.environ["ORCHESTRATOR_DB_PASSWORD"]
    dsn = f"postgresql://{user}:{password}@{host}:{port}/{name}"

    # Stage 1-5: Wait for pipeline to reach HITL Gate 1 or auto-advance past it
    print("  Stages 1-5: Waiting for Validator → Classification → Q&A Gen → Quality → HITL...")
    phase = await _wait_for_phases(dsn, workflow_id,
        ["human_review", "ready_for_distribution", "assessment_active", "terminated", "completed", "failed"],
        TIMEOUT)

    if phase == "terminated":
        print("  Workflow terminated (e.g., insufficient material or content safety failure)")
        return phase
    if phase in ("completed", "failed"):
        print(f"  Workflow ended early: {phase}")
        return phase

    # Stage 6: HITL Gate 1 — Auto-approve questions
    # Always call approve — the Orchestrator may auto-advance past human_review
    # but the API Server's approve endpoint updates DB state needed for distribute.
    print("  Stage 6: Auto-approving questions (HITL Gate 1)...")
    try:
        approve_resp = _api("POST", f"/api/v1/assessments/{workflow_id}/approve", {"rejected_question_ids": []})
        print(f"  Approved: {approve_resp.get('status', 'ok')}")
    except Exception as e:
        print(f"  Approve call: {e} (may already be approved)")

    # Stage 7: Distribute to participants
    # Always call distribute — even if events show invitations already sent,
    # because the Orchestrator may auto-distribute without updating
    # assessment_participants.invitation_status to 'sent'.
    # The distribute endpoint is idempotent and updates participant status.
    invitations = []
    print("  Stage 7: Distributing assessment...")
    try:
        dist_resp = _api("POST", f"/api/v1/assessments/{workflow_id}/distribute", {})
        invitations = dist_resp.get("invitations", [])
        print(f"  Distributed to {len(invitations)} participant(s)")
    except Exception as e:
        print(f"  Distribute call failed: {e} (may need manual status update)")

    # If no invitations from distribute call, query assessment_participants DB table.
    # Token = participant UUID (E2E pattern). The event payload is often null.
    if not invitations:
        try:
            import asyncpg
            conn = await asyncpg.connect(dsn)
            try:
                rows = await conn.fetch(
                    "SELECT id, email FROM assessment_participants WHERE assessment_id = $1",
                    assessment_id,
                )
                invitations = [{"token": str(r["id"]), "email": r["email"]} for r in rows]
                print(f"  Fetched {len(invitations)} participant(s) from assessment_participants")
            finally:
                await conn.close()
        except Exception as e:
            print(f"  Could not fetch participants from DB: {e}")

    # Stage 8: Participants take assessment and submit
    print("  Stage 8: Auto-submitting participant answers...")
    for inv in invitations:
        token = inv.get("token", "")
        try:
            # Get questions
            q_resp = _api("GET", f"/api/v1/participate/{token}/questions")
            questions = q_resp.get("questions", [])
            print(f"  Participant {inv.get('email', '?')}: {len(questions)} questions")

            # Auto-answer: MCQ picks first option, OE gives substantive paragraph
            answers = []
            for q in questions:
                if q.get("type") == "mcq":
                    options = q.get("options", [])
                    answers.append({"question_id": q["id"], "answer_content": options[0]["label"] if options else "A"})
                else:
                    answers.append({
                        "question_id": q["id"],
                        "answer_content": (
                            "This concept involves collaborative software development practices "
                            "that improve code quality through continuous peer review. Key benefits "
                            "include reduced defects, knowledge sharing across the team, and improved "
                            "design decisions through real-time discussion. The practice requires "
                            "effective communication and mutual respect between participants."
                        ),
                    })

            # Submit
            submit_resp = _api("POST", f"/api/v1/participate/{token}/submit", {"answers": answers})
            print(f"  Submitted: {submit_resp.get('status', 'ok')}")
        except Exception as e:
            print(f"  Submit failed for {inv.get('email', '?')}: {e}")

    # Stage 9-10: Wait for grading + reporting.
    # The Orchestrator's StubWorkflowStoreAdapter doesn't write reporting/distribution
    # events to workflow_events, so event-based detection won't find them.
    # Use a bounded wait: 180s should be enough for grading (2 OE) + reporting.
    # If the pipeline hasn't emitted report_review/completed events by then,
    # proceed anyway — the Langfuse observations will be there.
    print("  Stages 9-10: Waiting for Evaluator (grading) → Reporting (max 180s)...")
    phase = await _wait_for_phases(dsn, workflow_id,
        ["report_review", "completed", "terminated"],
        180)  # bounded wait — reporting events may not be in workflow_events

    if phase == "report_review":
        # Stage 11: HITL Gate 2 — Distribute reports
        print("  Stage 11: Auto-distributing reports (HITL Gate 2)...")
        try:
            report_dist_resp = _api("POST", f"/api/v1/assessments/{workflow_id}/reports/distribute", {})
            print(f"  Reports distributed: {report_dist_resp.get('status', 'ok')}")
        except Exception as e:
            print(f"  Report distribute failed: {e}")

        # Wait for completion
        phase = await _wait_for_phases(dsn, workflow_id, ["completed"], 30)

    print(f"  Final phase: {phase}")
    return phase


def _evaluate_agent(agent: str, generations: list[dict]) -> dict[str, float]:
    """Run per-agent GEval metrics on Langfuse trace generations.

    Each agent gets criteria tailored to its domain — not generic metrics.
    The judge LLM evaluates the quality of actual prompt/response pairs
    from the workflow run.
    """
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams

    # Per-agent trace evaluation criteria — tightened to each agent's actual I/O patterns.
    # The Gemini judge reads: (input prompt, LLM response) from Langfuse + these criteria.
    agent_criteria: dict[str, dict[str, str]] = {
        "validator": {
            "safety_reasoning": (
                "This is a content safety synthesizer that merges multiple analyzer reports into a single "
                "structured response. The INPUT is a collection of analyzer findings (copyright notices, "
                "content flags, readability results). The OUTPUT is a Pydantic JSON response serialized "
                "from a Python model with 'discarded' and 'findings' arrays. This JSON format is the "
                "CORRECT and EXPECTED output format — it is NOT an error or malformed response.\n\n"
                "Scoring rules (apply in order):\n"
                "1. Is the output valid structured JSON with 'discarded' and/or 'findings' fields? "
                "If yes: score at least 0.8.\n"
                "2. Does the output address analyzer findings from the input (copyright notices, content "
                "flags)? If yes: score at least 0.9.\n"
                "3. Does the output avoid fabricating issues not present in the input? If yes: score 1.0.\n\n"
                "CRITICAL: Educational material from universities (e.g., NUS 'All Rights Reserved') will "
                "have copyright notices in the analyzer input. Listing these as informational findings in "
                "the output is CORRECT behavior — do NOT penalize it. A well-formed JSON response with "
                "copyright findings from educational content should score 0.9 to 1.0."
            ),
        },
        "validator-moderation": {
            "moderation_accuracy": (
                "This is an OpenAI Moderation API result for content pre-screening. "
                "The input is text content sent to the moderation endpoint. "
                "The output contains category flags and confidence scores. Score strictly:\n"
                "1. FALSE POSITIVE CHECK: Educational content about software engineering, pair programming, "
                "or assessment methodology must NOT be flagged as harmful. Any false positive on standard "
                "academic content scores 0.\n"
                "2. CATEGORY APPROPRIATENESS: Flagged categories must match the actual content. Flagging "
                "'violence' on programming content is wrong. Each misclassified category reduces the score.\n"
                "3. CONFIDENCE CALIBRATION: Confidence scores should be near 0 for safe educational content "
                "and high only when genuinely problematic content is present.\n"
                "4. PII AWARENESS: If the content contains email addresses, phone numbers, or identity "
                "numbers (e.g., NRIC), the moderation pipeline should detect them."
            ),
        },
        "classification": {
            "sufficiency_quality": (
                "This is a Classification Agent LLM call that assesses material sufficiency, extracts topics, "
                "or evaluates rubric fitness. The input contains document chunks and assessment configuration "
                "(question count, difficulty level). Score strictly:\n"
                "1. VERDICT-EVIDENCE ALIGNMENT: If the output says 'sufficient', the input chunks must "
                "demonstrably contain enough depth for the requested question count and difficulty. "
                "Claiming sufficiency for 30 hard questions from 3 shallow paragraphs scores 0.\n"
                "2. TOPIC TRACEABILITY: Each extracted topic must correspond to actual content in the input "
                "chunks. Topics that don't appear in any chunk are hallucinated and score 0.\n"
                "3. GAP SPECIFICITY: If insufficient, the gap analysis must name specific missing domains "
                "based on what the chunks DON'T cover, not generic gaps. Vague gaps like 'needs more content' "
                "without specifying what content score below 0.5.\n"
                "4. RUBRIC COHERENCE: If assessing rubric fitness, the alignment score must reflect actual "
                "semantic overlap between the rubric criteria and the chunk content. Claiming alignment "
                "when the rubric covers a different domain than the material scores 0.\n"
                "5. AUTONOMY LOGIC: If web_research_mode is 'auto' and material is insufficient, the output "
                "must include search queries that target the identified gaps. Queries unrelated to the gaps "
                "score below 0.5."
            ),
        },
        "qna-generation": {
            "question_relevance": (
                "This is a Q&A Generation Agent LLM call that produces assessment questions from source material. "
                "The input contains extracted topics, document chunks, and generation parameters (MCQ count, "
                "OE count, difficulty). Score strictly:\n"
                "1. ANSWERABILITY FROM SOURCE: Every question must be answerable using ONLY the source chunks "
                "provided in the pipeline (not general knowledge). A question about a topic not in the "
                "material scores 0. Check: could a student answer this from the provided text alone?\n"
                "2. DIFFICULTY CALIBRATION: 'Hard' questions should require synthesis across multiple chunks, "
                "not simple recall. 'Easy' questions should test basic comprehension. Mismatched difficulty "
                "scores below 0.5.\n"
                "3. MCQ DISTRACTOR QUALITY: Each MCQ must have distractors that are plausible to someone "
                "who partially understands the topic. Obviously wrong distractors (e.g., random words, "
                "different domain entirely) score below 0.3.\n"
                "4. TOPIC DISTRIBUTION: Questions should span the extracted topics, not cluster on one area. "
                "If 5 topics exist but all questions are about topic 1, score below 0.5.\n"
                "5. MODEL ANSWER GROUNDING: For open-ended questions, the model answer must reference "
                "specific concepts from the source chunks, not provide generic textbook answers."
            ),
        },
        "evaluator": {
            "grading_quality": (
                "This is an Evaluator Agent LLM call that grades participant submissions or validates "
                "question quality. The input contains the question, the participant's answer (or generated "
                "Q&A for quality validation), and optionally rubric criteria. Score strictly:\n"
                "1. RATIONALE SPECIFICITY: The grading rationale must reference the actual content of the "
                "participant's answer, not provide generic feedback. 'Good answer' without citing what was "
                "good scores 0.\n"
                "2. RUBRIC ADHERENCE: If rubric criteria are provided, the score must align with the rubric's "
                "marking scheme. Awarding full marks when the answer misses rubric criteria, or zero marks "
                "when the answer addresses them, scores 0.\n"
                "3. SCORE PROPORTIONALITY: Partial answers should receive partial credit. Binary scoring "
                "(full or zero) for nuanced open-ended responses scores below 0.5.\n"
                "4. NO FABRICATION: The rationale must not invent rubric criteria or answer content that "
                "isn't present in the input. Each fabricated reference reduces the score.\n"
                "5. MCQ CORRECTNESS: For MCQ grading, the deterministic check must match the correct answer "
                "key. Any MCQ scoring error scores 0."
            ),
        },
        "web-research": {
            "research_relevance": (
                "This is a Web Research Agent LLM call that decomposes search queries or summarizes web "
                "search results to fill identified material gaps. The input contains gap topics from the "
                "Classification Agent and search parameters. Score strictly:\n"
                "1. QUERY-GAP ALIGNMENT: Each generated sub-query must directly target an identified "
                "content gap. Queries about unrelated topics score 0.\n"
                "2. SUMMARY ACCURACY: Summarized search results must accurately represent the source "
                "content. Fabricated facts or statistics score 0.\n"
                "3. EDUCATIONAL VALUE: The synthesized content must be suitable for generating assessment "
                "questions at the specified difficulty level. Superficial summaries that lack assessable "
                "depth score below 0.5.\n"
                "4. SOURCE HYGIENE: The output should not include promotional content, ads, or irrelevant "
                "web noise. Each instance of unfiltered noise reduces the score."
            ),
        },
        "reporting": {
            "report_quality": (
                "This is a Reporting Agent LLM call that generates a participant feedback report from "
                "evaluation results. The input contains graded submissions, scores per question, and "
                "optionally rubric criteria. Score strictly:\n"
                "1. SCORE ACCURACY: Every score mentioned in the report must exactly match the evaluation "
                "results in the input. Inventing or rounding scores differently scores 0.\n"
                "2. FEEDBACK TRACEABILITY: Per-question feedback must reference the specific question "
                "content and the participant's actual answer. Generic feedback like 'try harder' without "
                "citing the question scores 0.\n"
                "3. MCQ EXPLANATION: For wrong MCQ answers, the report must explain why the chosen option "
                "is incorrect AND why the correct option is right, using the pre-generated option "
                "explanations if available.\n"
                "4. TOPIC GAP EVIDENCE: Topic weakness analysis must be grounded in actual wrong answers, "
                "not assumed. Claiming 'weak in algorithms' when the participant answered all algorithm "
                "questions correctly scores 0.\n"
                "5. CONSTRUCTIVE TONE: The report should be professional and encouraging, with actionable "
                "improvement suggestions. Harsh or discouraging language scores below 0.5."
            ),
        },
    }

    criteria_map = agent_criteria.get(agent, {
        "general_quality": (
            "Evaluate whether this LLM output is relevant to the input prompt, "
            "well-structured, and free of hallucinations or fabricated information."
        ),
    })

    all_scores: dict[str, list[float]] = {}

    for g in generations:
        # Serialize dicts/lists as proper JSON (not Python repr with single quotes)
        raw_inp = g.get("input", "")
        raw_out = g.get("output", "")
        inp = json.dumps(raw_inp, indent=2) if isinstance(raw_inp, (dict, list)) else str(raw_inp)
        out = json.dumps(raw_out, indent=2) if isinstance(raw_out, (dict, list)) else str(raw_out)
        if len(inp) < 20 or len(out) < 10:
            continue

        tc = LLMTestCase(input=inp, actual_output=out)

        for metric_key, criteria_text in criteria_map.items():
            if metric_key not in all_scores:
                all_scores[metric_key] = []
            try:
                config: dict = {
                    "name": metric_key,
                    "criteria": criteria_text,
                    "evaluation_params": [
                        LLMTestCaseParams.INPUT,
                        LLMTestCaseParams.ACTUAL_OUTPUT,
                    ],
                    "threshold": 0.7,
                }
                if JUDGE_MODEL:
                    config["model"] = JUDGE_MODEL
                metric = GEval(**config)
                metric.measure(tc)
                if metric.score is not None:
                    all_scores[metric_key].append(metric.score)
                    status = "PASS" if metric.score >= 0.7 else "WARN"
                    print(f"    {metric_key}: {metric.score:.4f} ({status})")
            except Exception as e:
                print(f"    {metric_key}: ERROR — {e}")

    result: dict[str, float] = {}
    for metric_key, vals in all_scores.items():
        if vals:
            result[metric_key] = round(sum(vals) / len(vals), 4)
    return result


def _match_agent_obs(obs: dict, agent: str) -> bool:
    """Check if a Langfuse observation belongs to an agent.

    Primary: metadata.agent_name (set by LangfuseTracingAdapter).
    Fallback: observation name substring matching via AGENT_PATTERNS.
    """
    # Primary: check metadata.agent_name
    meta = obs.get("metadata") or {}
    meta_agent = meta.get("agent_name", "")
    # Expected patterns: "evaluator-agent", "reporting-agent", etc.
    agent_slug = f"{agent}-agent"
    if meta_agent == agent_slug or meta_agent == agent:
        return True
    # Fallback: observation name substring
    name_lower = (obs.get("name") or "").lower()
    return any(pattern in name_lower for pattern in AGENT_PATTERNS.get(agent, []))


async def run():
    print("=" * 70)
    print(f"  Golden Workflow Evaluation — {SCENARIO} scenario")
    print(f"  API Server: {API_SERVER}")
    print(f"  Langfuse: {LANGFUSE_HOST}")
    print(f"  Judge model: {JUDGE_MODEL}")
    print("=" * 70)
    print()

    config = SCENARIOS[SCENARIO]

    # Step 1: Create assessment
    print("Step 1: Creating assessment...")
    assessment = _api("POST", "/api/v1/assessments", config)
    assessment_id = assessment["id"]
    print(f"  Assessment: {assessment_id}")

    # Step 2: Upload material + rubric (download from GCS, then upload to API server)
    print("Step 2: Downloading material from GCS...")
    from google.cloud import storage as gcs_client
    client = gcs_client.Client()
    # GCS_BUCKET REQUIRED — silent default would point at the wrong bucket
    # and silently fail every download.
    bucket_name = os.environ["GCS_BUCKET"]
    gcs_bucket = client.bucket(bucket_name)

    # Material paths per scenario
    material_paths = {
        "insufficient": ("golden/eval-materials/insufficient/material.pdf", "golden/eval-materials/insufficient/rubric.pdf"),
        "sufficient": ("golden/eval-materials/sufficient/material.pdf", "golden/eval-materials/sufficient/rubric.pdf"),
    }
    material_gcs, rubric_gcs = material_paths.get(SCENARIO, material_paths["insufficient"])

    # Download material PDF
    material_content = gcs_bucket.blob(material_gcs).download_as_bytes()
    print(f"  Downloaded material ({len(material_content)} bytes)")

    # Upload material to API server
    def _upload_file(endpoint: str, filename: str, content: bytes, content_type: str = "application/pdf"):
        boundary = "----GoldenEvalBoundary"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode() + content + f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(endpoint, data=body, method="POST")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())

    material_resp = _upload_file(
        f"{API_SERVER}/api/v1/assessments/{assessment_id}/materials",
        "golden-material.pdf", material_content,
    )
    print(f"  Material uploaded: {material_resp.get('id', 'ok')}")

    # Upload rubric
    try:
        rubric_content = gcs_bucket.blob(rubric_gcs).download_as_bytes()
        print(f"  Downloaded rubric ({len(rubric_content)} bytes)")
        rubric_resp = _upload_file(
            f"{API_SERVER}/api/v1/assessments/{assessment_id}/rubrics",
            "golden-rubric.pdf", rubric_content,
        )
        print(f"  Rubric uploaded: {rubric_resp.get('id', 'ok')}")
    except Exception as e:
        print(f"  Rubric upload skipped: {e}")

    # Step 3: Start workflow
    print("Step 3: Starting workflow...")
    start_resp = _api("POST", f"/api/v1/assessments/{assessment_id}/start")
    workflow_id = start_resp.get("workflow_id", "unknown")
    print(f"  Workflow: {workflow_id}")

    # Step 4: Drive through full 13-stage pipeline (auto-process HITL gates)
    print(f"Step 4: Running full pipeline (timeout {TIMEOUT}s per stage)...")
    final_phase = await _run_full_pipeline(workflow_id, assessment_id)
    print(f"  Pipeline complete: {final_phase}")

    # Step 5: Fetch Langfuse traces
    print("Step 5: Fetching Langfuse traces...")
    # Wait for Langfuse to ingest (Model Broker fire-and-forget traces for
    # later-stage agents like Evaluator/Reporting need time to flush)
    await asyncio.sleep(30)

    traces = _langfuse_get("/api/public/traces?limit=10")
    workflow_trace = None
    for t in traces.get("data", []):
        meta = t.get("metadata") or {}
        if meta.get("workflow_id") == workflow_id:
            workflow_trace = t
            break

    if not workflow_trace:
        print(f"  WARNING: No Langfuse trace found for {workflow_id}")
        print("  Trying by name match...")
        for t in traces.get("data", []):
            if workflow_id in (t.get("name") or ""):
                workflow_trace = t
                break

    if not workflow_trace:
        print(f"  ERROR: Cannot find trace for workflow {workflow_id}")
        print(json.dumps({"error": "trace_not_found", "workflow_id": workflow_id}))
        sys.exit(1)

    trace_id = workflow_trace["id"]
    print(f"  Trace: {trace_id}")

    # Fetch all observations (paginated — Langfuse caps at 100 per page)
    all_obs: list[dict] = []
    page = 1
    while True:
        obs_data = _langfuse_get(f"/api/public/observations?traceId={trace_id}&limit=100&page={page}")
        page_obs = obs_data.get("data", [])
        all_obs.extend(page_obs)
        total_items = obs_data.get("meta", {}).get("totalItems", len(page_obs))
        if len(all_obs) >= total_items or not page_obs:
            break
        page += 1
    generations = [o for o in all_obs if o.get("type") == "GENERATION"]
    print(f"  Observations: {len(all_obs)}, Generations: {len(generations)}")

    # Step 6: Evaluate per agent
    print()
    print("Step 6: Per-agent evaluation...")
    all_baselines: dict[str, dict] = {}

    for agent in AGENT_METRICS:
        agent_gens = [g for g in generations if _match_agent_obs(g, agent)]
        print(f"\n  ── {agent} ({len(agent_gens)} generations) ──")

        if not agent_gens:
            print(f"    No generations found (agent may not have run in {SCENARIO} scenario)")
            all_baselines[agent] = {"_no_data": True, "_reason": f"No generations in {SCENARIO} workflow"}
            continue

        scores = _evaluate_agent(agent, agent_gens)
        scores["_workflow_id"] = workflow_id
        scores["_scenario"] = SCENARIO
        scores["_generations_evaluated"] = len(agent_gens)
        scores["_total_observations"] = len(all_obs)
        all_baselines[agent] = scores

        for metric, val in scores.items():
            if not metric.startswith("_"):
                print(f"    {metric}: {val:.4f}")

    # Step 7: Write per-agent baselines
    print()
    print("Step 7: Writing baselines...")
    for agent, scores in all_baselines.items():
        baseline_path = BASELINE_DIR / f"baseline-{agent}.json"
        baseline_path.write_text(json.dumps(scores, indent=2))
        print(f"  Written: {baseline_path}")

    # Summary
    print()
    print("=" * 70)
    print(f"  Workflow: {workflow_id}")
    print(f"  Scenario: {SCENARIO}")
    print(f"  Final phase: {final_phase}")
    print(f"  Agents evaluated: {len([a for a in all_baselines if not all_baselines[a].get('_no_data')])}")
    print(f"  Agents with no data: {len([a for a in all_baselines if all_baselines[a].get('_no_data')])}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run())
