# Reproduction scope and execution contract

## Frozen submitted inference

Use the original container linked in the README. It was submitted for Linux/amd64 CPU-only inference with 16 GB platform RAM. The archived local serial benchmark used 2 CPUs and 7 GiB RAM; 772 inputs averaged 15.469 seconds each. These settings describe inference, not training.

To run the frozen image, obtain an authorised copy of its original complete model resource, unpack it under a model directory, and mount that directory read-only at `/opt/ml/model`. Mount a challenge-shaped case read-only at `/input` and a fresh output directory at `/output`. The entrypoint validates the model files against pinned hashes.

Input paths accepted by the preserved runtime include upper/lower IOS sockets under `files/ios-upper` and `files/ios-lower`, and photographs under `images/2d-intraoral-photographs` or `images/intraoral-photo`. See the preserved input-discovery code for exact supported alternatives. The output is `diagnostic-imaging-report.json` containing a nonempty `report` string.

The original image config digest is `sha256:eb4ec2d8206fe28057bfde9f5f05d3f7491b763d636ca1fb14b58fb00d5079df`. A source rebuild is proposed in `source_snapshot/Dockerfile.rebuild`; it has not been built or verified and is not the original submitted image.

## Task-specific training

Existing classifier-training functions are included in `source_snapshot/vlm_lite/photo_residual.py` and `source_snapshot/baselines/vc_fulltrain_model.py`. Exact training settings are in `METHOD_ACTUAL.md`. The frozen image feature encoder is not fine-tuned.

A complete reproduction additionally needs the original or reconstructed official-data preprocessing, split and duplicate grouping, nested out-of-fold geometry predictions, feature-cache manifests, report-parser labels, final fitting orchestration, mesh index construction and sentence-pool construction. These resources are not replaced by the compact coefficient export.

The original runners bind these resources by hashes and use project-specific experiment manifests. Copying a runner without its valid resource-generation chain does not establish reproducibility. Those orchestration and resource-generation steps must be prepared and independently verified before marking the survey README question Yes.

## Included and absent resources

The source snapshot and active fitted coefficients are included. The original official training data, reports, sentence pool, individual-case retrieval descriptors, split/case lists and upstream pretrained checkpoint are absent. Users must acquire authorised official data and upstream weights under their own terms. No training total wall time or exact historical training CPU/RAM specification is asserted.
