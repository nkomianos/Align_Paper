# ID-only attribution audit

The original four mechanisms always assign IDs consistently with chronology.
The broader follow-up changed multiple factors and cannot isolate this feature.
This audit reuses the original exact prompts, rule, generation implementation,
192token cap and scoring, with24original exposed base cases at presentation0.
Each is run twice: identity IDs and reversed IDs. Values and all edge endpoints
are remapped together; record positions and all other wording are unchanged.
Gold answers must be invariant.96calls total, direct and extract for48inputs.

Primary endpoint: paired change in semantic extraction under renaming. Also
report action accuracy and direct-answer changes, parse/EOS, and each mechanism.
No best-case selection or confidence interval treating48views as independent.
This is posthoc attribution, not confirmation or an encoding repair. A drop
supports ID sensitivity on these cases, not universal incapacity. No drop
means the broader failure cannot be attributed to this ID reversal alone.
The inherited verifier's old route field is not an admission decision for this
study; no training, Nemo or third prompt repair follows automatically.

CPU prerequisite: all48transformed gold answers agree with originals and
inverse renaming restores exact input text, values and edges. Full source and
wrapper hashes accompany output. Expected192token ceiling per call unchanged;
prior AWS192call measurements197–350seconds suggest a few minutes for96calls,
with timing confirmed from this run rather than assumed identical.
