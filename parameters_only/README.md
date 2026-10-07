# Active parameter export

This export was prepared from the model resource used by the actual Bite2Text Finding-Gated Retrieval / attempt05 submission. It contains no newly trained model and introduces no coefficient changes.

## Contents

- `geometry_parameters.json`: active geometry classifier heads, aggregate class counts used by frequency heads, 112-D feature standardisation and missing-feature policy.
- `photo_residual_parameters.json`: the 11 active photo residual heads, class order, feature standardisation and fusion policy.
- `dinov2_reference.json`: upstream DINOv2 checkpoint identifier, pinned revision and preprocessing contract. The upstream checkpoint is not copied.
- `parameter_counts.json`: active fitted and frozen parameter counts, with the unused legacy classifier excluded.

The extracted active arrays were compared element-by-element with the original loaded JSON values and match exactly. The active fitted total is 67,902: 3,390 geometry coefficients/biases and 64,512 photo-residual weights. DINOv2's 22,056,576 parameters remain frozen.

## Deliberately absent resources

No original training reports, report fragments, sentence pool, individual case-ID list, patient-level mesh retrieval index, image, mesh or credential is included. Aggregate class counts and standardisation statistics are retained as model configuration.

The original container validates exact original model files and hashes. These compact exports use a new documentation schema, so they are not directly loadable in place of the original model archive. An exact reproduction also needs the authorised retrieval resources and their original ordering. This export is for parameter inspection and preparation of a reviewed public release; it is not a standalone inference model.

## Licence

No new licence has been selected or applied to this export. The upstream DINOv2 checkpoint remains subject to its original terms. Its source is [facebook/dinov2-small](https://huggingface.co/facebook/dinov2-small).
