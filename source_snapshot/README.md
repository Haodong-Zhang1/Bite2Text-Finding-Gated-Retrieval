# Current Bite2Text attempt05 source snapshot

This folder contains a byte-preserving snapshot of the current implemented final attempt05 runtime source and selected existing training core functions. It does not implement the planned Qwen, PointNet++ or concept-query architecture.

## Inference source

The submitted entrypoint is `inference.py`, a 20,128-byte Python ZIP app copied byte-for-byte from the final submitted container. Its two internal source members are `__main__.py` and `tooth_finding_gate_v1.py`. They match `submission/bitevlm_lite_v3_attempt05/__main__.py` and `vlm_lite/tooth_finding_gate_v1.py` byte-for-byte. Its project-owned Python dependency closure is included. The implementation combines the frozen BiteVLM-Lite V2 runtime, fact reranking, a tooth-finding gate and per-case fail-open handling. The frozen DINOv2 feature encoder is included as code; pretrained weights are not included.

The ZIP app imports `tooth_finding_gate_v1` from its own internal source member. The original source is `vlm_lite/tooth_finding_gate_v1.py`; an identical root-level copy is also retained for the unpacked developer layout. `SOURCE_MANIFEST.json` records both source copies and ZIP-member hashes. Packages that originally use Python namespace packages remain namespace packages.

The source dependency inventory is in `DEPENDENCY_AUDIT.json`. It was derived recursively from the AST for the entrypoint and the selected training core. All project-owned imports resolve in this snapshot. No dynamic Python import calls were found in the included source. Runtime reading of model/data files is a separate dependency and is not covered by Python import resolution.

## Required model resources

The runtime requires four resource directories under `/opt/ml/model`: `bitevlm_lite_v2`, `vc_global_policy_v1`, `mesh_retrieval_v1` and `dinov2-small-ed25f3a`. Exact member names and expected SHA-256 values are retained in `submission/bitevlm_lite_v2/inference.py`.

No trained parameter files, pretrained weights, reference reports, sentence pools, case lists, meshes, photographs, or evaluation outputs are copied here. In particular, the retrieval reports and sentence pool require a redistribution review before public release. Source alone cannot run final inference without the matching authorised resources.

## Training core scope

`vlm_lite/photo_residual.py` contains the implemented deterministic linear residual-head fitting routine and feature/prediction alignment utilities. `baselines/vc_fulltrain_model.py` contains the full-training geometry field model fitting and serialization routines. The mesh descriptor and parser used by the runtime are also present.

This is a training-core snapshot, not an end-to-end reproducibility release. It omits the original experiment orchestration, nested cross-validation runner/protocols, cached inputs, duplicate-group/case split manifests, report-derived resources, training outputs and checkpoint bundles. No claim is made that these files alone reproduce the submission weights or every reported experiment.

## Dependencies and container references

`submission/bitevlm_lite_v2/requirements.txt` is the unchanged frozen CPU inference dependency list. Root `requirements.txt` is the unchanged original project dependency list and supplies additional training dependencies such as SciPy and scikit-learn. No environment was created and no package was installed during preparation.

`provenance/Dockerfile.v3.reference` is an unchanged historical v3 Dockerfile. Its entrypoint is older than attempt05 and it is retained as provenance only; it is not a complete final build definition.

`Dockerfile.rebuild` is newly written to run the archive-derived ZIP app with `python inference.py` and this source snapshot. It is an unbuilt rebuild proposal, not the submitted container and not a verified byte-for-byte reconstruction. The final submitted container archive must be shared separately. The rebuild expects authorised model resources to be mounted separately at `/opt/ml/model` and challenge inputs and outputs at `/input` and `/output`.

## Validation boundary

Preparation performed byte/hash equality checks, recursive project import resolution, AST syntax parsing and static privacy screening. A read-only streaming comparison applied all 17 final-container layers in manifest order, including relevant whiteout rules. All 15 standalone compared source/dependency files matched; both ZIP-app source members also matched. The raw ZIP entrypoint hash matches the final container. There was no source-version drift among these checked files; the initial entrypoint mismatch was a ZIP-versus-unpacked layout difference. See `../SOURCE_CONTAINER_COMPARISON.json`. No model, training job, container build, inference benchmark, scientific evaluation, or dependency installation was run. `VALIDATION.json` reports static checks and counts without printing matched private content.

No new project licence is asserted by this snapshot. Applicable licensing must be resolved before claiming an open-source licence or post-challenge publication eligibility.
