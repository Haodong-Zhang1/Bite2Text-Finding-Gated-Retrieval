# Bite2Text Finding-Gated Retrieval

Public code and active-parameter release for the implemented final attempt05 submission to ODIN 2026 Bite2Text. The algorithm combines a deterministic paired-mesh descriptor, field classifiers, a frozen DINOv2 photograph encoder, linear residual heads, report retrieval and controlled sentence editing.

## Public downloads

- [Original final submitted container](https://drive.google.com/uc?export=download&id=1ZoY3oYOT79tleW3ad-EQH78ttnwfm2pr)
- [Source and parameter review archive](https://drive.google.com/uc?export=download&id=1w3o1eSM0s3uz-UXHYtbSMlYaaUV_jqLn)
- [Active fitted parameter export](https://drive.google.com/uc?export=download&id=1qBNf_0bt8wSqSSTkLdciaW7sdVjJ4oVz)

All three complete anonymous downloads were compared with their original SHA-256 values. The container archive SHA-256 is `e1b249b7325c5e0f74fcad5256c0250c510bcccc546ce9ba39ffda3b8e4c3182`.

## Contents

- `source_snapshot/`: final runtime source, original ZIP-app entrypoint, selected classifier-training functions, dependency lists and static validation.
- `parameters_only/`: 67,902 active fitted geometry/photo-residual coefficients plus preprocessing metadata. This export uses an inspection schema.
- `METHOD_ACTUAL.md`: implemented architecture, training procedure and measured resource information.
- `SOURCE_CONTAINER_COMPARISON.json`: source-versus-final-container byte comparison.
- `REPRODUCIBILITY.md`: execution contract and current reproduction boundary.
- `LICENSE_STATUS.md`: current licensing status; no new project licence is applied.

## Inference requirements

The original container expects challenge-shaped inputs under `/input`, an output directory under `/output` and the original matching model resource under `/opt/ml/model`. The model root requires `bitevlm_lite_v2/`, `vc_global_policy_v1/`, `mesh_retrieval_v1/` and `dinov2-small-ed25f3a/`.

The complete original model resource is not distributed here. It contains official training-reference reports, a sentence pool and a case-level retrieval index. Redistribution terms remain unresolved. The extracted parameter JSON files cannot replace its original hash-validated files.

DINOv2 is referenced as `facebook/dinov2-small`, pinned to revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`. Its [upstream checkpoint](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056) is publicly available under Apache-2.0. The repository's licensing status does not alter upstream terms.

## Current reproduction status

This release includes the preserved inference source and active fitted parameters. Complete training reproduction has not been verified. The original training runners depend on hash-bound protocols, splits, labels and feature-cache manifests that are not included. The current survey answer for a README sufficient to reproduce all experiments remains **Not yet**.

No training, container rebuild or independent inference reproduction was run during this preparation. Original source and active parameter arrays were preserved. See `source_snapshot/README.md` for the checked runtime closure and unbuilt rebuild proposal.
