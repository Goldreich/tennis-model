# Source configuration

`sources.yaml` is a strict YAML serialization of
`tennis_model.schemas.SourceManifest`. It pins the audited ATP and WTA yearly
objects by repository commit, Git blob, SHA-256, coverage, retrieval time,
license, date semantics, and normalization contract. The registered sources are
historical research inputs under CC BY-NC-SA 4.0; they are not a current-tour
feed and must not be represented as current US Open coverage.

Do not replace an immutable locator with a moving-branch URL, infer one tour's
coverage from the other, or claim that an observed terminal date is complete
without a new source audit and manifest version.

# Model configuration

`model_v1.yaml` contains every numerical prior, optimizer, curvature, context,
and diagnostic choice used by the Milestone 3 serve-component fits. Loading is
strict: missing/extra fields, duplicate YAML keys, nonfinite values, quoted
numeric/boolean coercions, and invalid bounds fail explicitly. Each fit embeds the
parsed configuration and its canonical SHA-256.

The prose specification does not assign numerical prior scales, shrinkage-scale
hyperpriors, concentration bounds, or a dense-curvature size threshold. The
checked-in values are therefore auditable implementation concretizations, not new
model features or specification-derived constants. Any proposed change that can
alter probabilities requires explicit framework-version review under `AGENTS.md`.
