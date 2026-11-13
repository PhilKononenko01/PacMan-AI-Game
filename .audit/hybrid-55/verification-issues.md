# Evidence verifier review findings

V-01. P1. Schedule length inferred from records allowed a shortened report to pass. Fixed by checking independently declared final seed slots, repeats, checkpoint slots, and replicate. Permanent missing-game and wrong-replicate negative controls added.

V-02. P1. Sampled replays could be labeled complete and archive default sampled only five rows. Fixed by making full replay the default and requiring every game to replay before archival. Sampling explicitly yields verified=false. Existing 17,000-game validation replays were full and remain supported.

V-03. P2. Report hashes and rules were compared with themselves rather than loaded policies. Fixed by binding report and every record to actual frozen table hashes, rules, feature versions, and registered replicate. Permanent malformed provenance controls added.

All three findings concerned new verification tooling, not changes to recorded game outcomes. Independent reviewer hybrid_evidence_review reproduced them on tiny temporary fixtures before any final archive was accepted.
