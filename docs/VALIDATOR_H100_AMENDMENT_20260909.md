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
