# Prompt C hypothesis check
17 October 2026. Completed in the current conversation; not the required new-chat AI Reviewer report.

Research question: Can distillation improve CV–job fit classification and can post-training quantization retain the improvement while reducing storage?
M0: hard-label BERT-Tiny. M1: matched BERT-Tiny with teacher distillation. M2: M1 with dynamic INT8 Linear modules and 8-bit weight-only Embedding modules.
F0/F1/F2 are macro-F1 on a 0–100 scale; S1/S2 are comparably serialized model sizes.

1. H1 compares M1 against M0: yes. F1 − F0 >= 1.0 percentage point.
2. H2 compares M2 against M1 and assesses M2 against M0: yes. S2 <= 0.40*S1; F1 − F2 <= 1.0 point; F2 − F0 >= 0.5*(F1 − F0), conditional on F1 > F0. A non-improving M1 fails the retention claim.
3. Testable: yes. Fixed splits, paired seeds, serialized bytes and a declared macro-F1 scale permit direct checks.
4. Numerical justification: exploratory. The 60% target is motivated by covering the dominant embedding table but has not been established by a pilot. Metadata and unquantized tensors prevent a guaranteed fourfold reduction.
5. Fair and feasible: planned safeguards include matched initializations, equal M0/M1 search and update budgets, validation-only selection, held-out test jobs and reporting teacher cost. Colab and quantization compatibility remain untested. Familiar CVs across splits restrict the generalization claim to new jobs.
6. Quality and efficiency: both included. Macro-F1/accuracy, model size, parameters, MACs/FLOPs, CPU latency and memory are planned. Size reduction does not imply a speedup.

Two important corrections/checks:
1. Run a Colab compatibility and size check before locking the 60% target. If using the fallback, change M2 and H2 together before test evaluation.
2. Confirm the actual teaching-week allocation. Weeks 10–15 are proposed, not calendar-verified; ensure the matched runs fit available free Colab time.

No readiness score or experimental result is assigned by this check. The official final reviewer step must still occur in a new chat after student revision.
