# Validator G0 H100 deployment amendment

Recorded before any validator model endpoint is collected. The user authorized
the Lambda H100 allocation; the old awaiting-authorization status is obsolete.

The launch script and config required 85,899,345,920 bytes (80 GiB). The available
nominal 80 GB H100 reports 81,559 MiB, so that prerequisite would reject it before
loading either model. Set the hardware minimum consistently to 80,000,000,000
bytes. Actual model instantiation and sampling remain mandatory. This changes
hardware admission, not any scientific effect, power, validity or kill threshold.

Python 3.12, torch 2.7.1/CUDA 12.8 and the pinned dependency versions are installed
in a separate environment. The prior experiments' runtime is not modified.
Regenerate the source/config hashes for the current committed source tree rather
than disabling integrity checks. The corpus, model pins, prompts, sampling,
eligibility, CWE split, controls, and analysis are unchanged.

Deployment started at 06:04 UTC under remote bootstrap PID 10530. The public pinned
Gemma snapshot downloaded successfully; environment installation is still in
progress at this entry. No validator result or runtime estimate has been measured.
The old 2–4 hour GH200 estimate is not an H100 measurement.

Environment installation finished at 06:06 UTC. The generic H100 lock pins
SymPy 1.13.3, conflicting with torch 2.7.1's requirement. The dedicated
requirements/validator-h100-cu128.lock preserves its other pins and uses SymPy
1.14.0. After applying it, uv pip check reports all 83 packages compatible.
The formal CPU verifier still requires its separate offline lock/environment.

Oracle preflight passed. Initial model preflight failed because the Qwen cache
lacked .gitattributes, LICENSE and README.md; completing the same pinned snapshot
resolved this. Resume then loaded Qwen but Gemma's AutoProcessor import required
torchvision, absent from the generic lock. Installed torchvision 0.22.1+cu128,
matching torch 2.7.1; all 84 installed packages pass dependency checking. Added
this requirement to the deployment lock for reproducibility. The executing
source checkout and RUN_BINDING remain unchanged; no experimental completions
were collected before either fix. Failed logs remain in the run's recovery
history. Second resume uses validator_resume2.log.

Both model smokes passed on the second resume. The launcher then failed parsing
its own preflight log: stderr weight-loading progress had been merged into JSON
stdout. Separate and preserve stderr in its own log, keeping stdout as the strict
JSON record. Preserve validator_g0_v1 unchanged. A new committed launch/root is
required because orchestration changed; no experimental generations existed in
v1. Oracle checks passed for 32 tasks and 64 mutants (47 plausible incomplete).

At 06:19 UTC, after v2 began patch generation, the broader CPU test suite exposed
a missed copy of the old 80 GiB floor in the independent verifier. Five verifier
tests failed at this configuration check before their intended assertions. The
two-line correction changes only the minimum-byte comparison and its error text;
all eight tests in test_validator_monoculture_verify.py then pass. No generated
patches or test outcomes were used to select this correction.

Do not mutate or restart the generating checkout. Use a separately committed
verifier with its own source/commit attestation. The dispatch helper requires
that its entire src diff from the generating commit is exactly this two-line
unit correction, otherwise it rejects the independent verifier. Original run
binding and artifact hashes remain required. The receipt distinguishes generation
and verifier identities. No scoring, controls, filtering or thresholds change.

## Unicode transport repair

v2 terminated at 07:04:59 UTC in Qwen patch-aware collection after one durable
record. Both 64-suite specification-only phases were complete. A parsed JSON
string contained a lone surrogate; the text writer attempted literal UTF-8 and
raised UnicodeEncodeError. Preserve the complete failed root (archive SHA256
fe9f8f45a2cdca7a750567fcebf925a7354651826f371b9af08ab21ee3ee65b2).

Repair canonical JSON transport to escape otherwise unencodable surrogate code
points, preserving parsed values rather than rejecting or dropping the suite.
Apply the same representation to JSON files and validator content hashing.
Normal Unicode serialization remains byte-identical. Regression checks cover
durable append/resume, JSON/JSONL round trips, literal backslash escapes, Unicode
keys and unchanged normal-Unicode commitments. Twenty-four targeted tests pass.

A fresh committed repair run will repeat the same frozen prompts, seeds, model
pins, corpus and gates. Do not splice source identities into v2 or count repeated
draws as independent replications. Compare overlapping outputs after completion;
software repair is not a new scientific condition. Preserve every earlier attempt.
