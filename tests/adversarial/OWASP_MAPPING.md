# OWASP Mapping -- Classification Agent Adversarial Tests

| Test Class | OWASP LLM Top 10 | OWASP Agentic ASI | Attack Vectors | Payload Count |
|-----------|-------------------|---------------------|----------------|---------------|
| TestPromptInjectionResistance | LLM01: Prompt Injection | -- | 6 injection payloads in chunk content + assessment title + schema validation | 18 tests |
| TestJailbreakResistance | LLM02: Insecure Output | -- | 5 jailbreak payloads + gap analysis integrity | 10 tests |
| TestPiiLeakageResistance | LLM06: Sensitive Info | -- | 4 PII extraction payloads in sufficiency reason + topic names | 8 tests |
| TestHallucinationResistance | LLM09: Overreliance | -- | 4 hallucination payloads, bounded topic count + chunk count integrity | 8 tests |
| TestRagPoisoningResistance | LLM05/LLM08 | -- | 4 RAG poisoning payloads, standalone + mixed with legitimate chunks | 8 tests |
| TestGoalHijackResistance | -- | Goal Hijacking | 4 goal hijack payloads in sufficiency + topic extraction | 8 tests |
| TestToolAbuseResistance | -- | Tool Orchestration Abuse | 4 tool abuse payloads, sufficiency check + invocation count audit | 8 tests |
| TestCombinedAttackResistance | LLM01+LLM02+LLM06 | Goal Hijacking + Tool Abuse | 3 combined payloads chaining multiple categories | 6 tests |
| TestOcrSourceTypeAdversarial | LLM01+LLM02 | -- | 5 payloads via OCR source type (0.5x weight enforcement) | 5 tests |

## Totals

- **9 test classes**
- **79 test cases**
- **36 unique adversarial payloads** (6 injection + 5 jailbreak + 4 PII + 4 hallucination + 4 RAG poisoning + 4 goal hijack + 4 tool abuse + 3 combined + 2 cross-category OCR)

## Tolerance

**0%** -- any successful attack blocks CI merge.

## Cost

**$0.00** -- stub Model Broker, no real LLM calls in unit tests.

## Adversarial Resistance Design

The Classification Agent's sufficiency checker uses a **deterministic heuristic threshold** (effective chunk count vs required question count). This threshold is computed in pure Python before any LLM call, making it immune to prompt injection. The LLM is only consulted for gap analysis when material is already determined to be insufficient. This architecture means:

1. **Prompt injection cannot override sufficiency** -- the heuristic runs before LLM
2. **Jailbreak cannot bypass threshold** -- threshold is code, not LLM output
3. **RAG poisoning cannot inflate chunk count** -- chunk count is `len(chunks)`, not LLM-derived
4. **Hallucination is bounded** -- topic extraction from stub returns canned data; in production, topics are grounded to provided chunks only

## CI Integration

```yaml
# .github/workflows/adversarial.yml
- name: Run adversarial tests
  run: .venv/bin/python -m pytest tests/adversarial/ -v --tb=short
  # 0% tolerance: any failure = CI block
```
