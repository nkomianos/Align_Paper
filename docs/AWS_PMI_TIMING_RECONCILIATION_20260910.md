# AWS PMI timing reconciliation

Five saved timing records match their original manifests. Their sum is
1,222.810 seconds, or20.380 minutes (0.339669 hours), of recorded program sections.

| Completed section | Seconds |
|---|---:|
| Nonthinking prefix diagnostic |101.769|
| Thinking trace generation |395.195|
| Initial full-prefix comparison, numerical qualification failed |21.723|
| Three-case numerical diagnosis |8.837|
| Corrected cached comparison |695.286|

These are wall-clock timers around different code sections, some including
model loading and CPU work. They are **not GPU-kernel hours**, billed instance
hours, a complete research-program total, or H200-equivalent hours. Untimed
preparation/loading, weight hashing, downloads, transfers, initial failed
launches and idle allocation are not covered by this sum. It would be incorrect
to subtract0.339669 from50 and claim the remainder as the available GPU budget.

The failed numerical comparison remains charged to this recorded work total;
excluding invalid assays from scientific evidence does not erase their cost.
The last AWS GPU process check at11:04UTC was empty. No later provider lifecycle
change has been performed by this task. A billing total requires provider
allocation records; no cost or credit balance is inferred here.

Reproduce with `python scripts/audit_pmi_timing_ledger.py`. Output is
`artifacts/pmi_prefix_diagnostic_20260910/TIMING_LEDGER.json`, including source
hashes and explicit unavailable fields. This is an accounting audit, not a
scientific result or an admission for another experiment.
