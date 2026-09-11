# Poison-complexity G0 v3 protocol

V2 passed clean and near-trigger qualification but failed the explicit
conditional-rule capability gate: raw accuracy was 59.38% for Qwen2.5-0.5B and
54.69% for Qwen2.5-1.5B. Inspection of aggregate outputs is consistent with a
numeric-switch interface problem. No V2 training or scientific endpoint ran.

V3 is the final permitted capability-interface repair. It replaces numeric
switch 0/1 with an explicit semantic selector, `LEFT` or `RIGHT`. The
conditional payload returns the digit on the selected side. Projection always
returns the left digit. Both have exactly matched digit marginals over every
complete 20-case block. All other frozen V2 settings and thresholds remain
unchanged.

If either model remains below 85% on the explicit-rule condition, the assay is
killed. No further prompt or task repair is admitted for this candidate.
