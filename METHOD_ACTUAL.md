# Bite2Text Finding-Gated Retrieval: actual submitted method

This document describes the implemented attempt05 inference system, which reuses the frozen V2 model resources. It does not describe the proposed future VLM architecture. The English answers below can be adapted to the challenge survey; the provenance notes should be omitted when pasting into the form.

## Method name or short identifier

Bite2Text Finding-Gated Retrieval (BiteVLM-Lite, attempt05).

## Method summary (200–400 words)

Bite2Text Finding-Gated Retrieval generates orthodontic reports from paired upper and lower intraoral surface scans and intraoral photographs using a lightweight combination of structured finding prediction and retrieval-based report editing. Its task-specific design uses low-capacity multimodal classifiers to guide the selection and revision of existing clinical wording, while keeping the visual encoder frozen.

Each pair of surface scans is represented by a deterministic 112-dimensional descriptor containing scan-level statistics and relative upper–lower information. A geometry baseline predicts 14 categorical clinical fields using field-specific logistic classifiers or training-frequency predictions. Photographs are encoded with a frozen DINOv2 ViT-S/14 backbone. Global and patch-averaged features are combined within each photograph, and mean/max pooling produces a permutation-invariant representation of the photo set. Separate linear residual heads update the geometry probabilities for 11 predefined fields. Dentition and the curves of Spee and Wilson retain geometry-only predictions. Missing or unusable photographs bypass the residual branch.

The report generator retrieves the five nearest training reports in standardized mesh-descriptor space. It compares their parsed findings with the predicted clinical fields and changes the nearest-neighbor donor only when a fixed mismatch-reduction rule is satisfied. It then replaces eligible sentences with exact-finding-matched sentences from a training-reference sentence pool. Parser-based checks reject replacements that alter unrelated structured fields.

The final attempt05 post-processing removes tooth-level crossbite localization and filters additional tooth-level pathology details using their support among the five retrieved neighbors. Retained pathology statements are reduced to generic wording where appropriate. This support is a retrieval-based consistency heuristic, rather than direct verification of a finding in the input images. A 240-second prediction deadline and a predefined fallback report handle per-case prediction errors or timeouts.

The approach is intended to combine complementary visual cues with limited trainable capacity and preserve dataset-specific reporting style. It does not provide a guarantee of clinical correctness: findings can be misclassified, retrieval can transfer inappropriate details, and the report parser does not capture every clinically relevant statement.

## Input handling and pre-processing

The 3D inputs are upper and lower intraoral surface meshes. No volumetric resampling or voxel-intensity normalization is performed. Each scan is deterministically sampled to at most 20,000 triangles. A 53-dimensional descriptor is computed for each scan using the centroid, extents, centered-coordinate and radial quantiles, principal scales, triangle-area and normal statistics, and log-transformed triangle-count and canonical-file-size terms. Concatenating the two descriptors with centroid differences and extent ratios gives 112 features. This uses the supplied mesh coordinates; it does not estimate a new tooth-level registration or segmentation. Descriptor standardization uses training-derived statistics.

Photographs are converted to RGB, resized with bicubic interpolation so that the shortest edge is 256 pixels, center-cropped to 224 × 224 pixels, rescaled by 1/255, and normalized with ImageNet channel means and standard deviations. The image encoder processes a fixed physical batch of 16 pages, padding when necessary and discarding padded outputs. Each page produces a 768-dimensional concatenation of its CLS token and mean patch token. Concatenated mean and maximum pooling over the available pages gives a 1,536-dimensional photo-set feature. Numeric filenames are not interpreted as clinical view labels. The 2D and 3D branches are combined at the level of clinical-field probabilities rather than through voxel fusion.

## Model architecture

The geometry component, G0, uses standardized 112-dimensional mesh descriptors and field-specific multinomial logistic regression or training-frequency predictions. The frozen image backbone is `facebook/dinov2-small`, DINOv2 ViT-S/14, at revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`. Its 384-dimensional token embeddings produce 768-dimensional page features and a 1,536-dimensional pooled set feature. Eleven independent no-intercept linear residual heads add photo-derived logits to the log geometry probabilities, followed by softmax and deterministic category selection.

The 14 structured fields are dentition, maxillary constriction, crossbite, vertical bite relationship, overjet, midline, the curves of Spee and Wilson, upper and lower crowding, and right and left molar and canine classes. The photo branch updates all except dentition, Spee, and Wilson. The text component consists of mesh-neighbor retrieval, finding-aware donor selection, parser-checked sentence replacement, and the final tooth-finding gate. No LLM, generative VLM decoder, learned tooth segmentation network, or learned point-cloud backbone is used by the submitted inference system.

## Training strategy

For the implemented concept-prediction stage, clinical labels were derived from the official training reports using the fixed structured-report parser. Development used five-fold internal out-of-fold evaluation over 772 training cases, with known duplicate components kept within the same fold. Patient-level independence cannot be claimed because suitable patient-identity metadata were unavailable. Strict nested cross-fitting generated geometry probabilities for residual training: the geometry model excluded both the current outer evaluation fold and each meta-training case's fold. Scaling and residual fitting used only the corresponding training cohort.

The geometry logistic models used L2-regularized multinomial logistic regression with L-BFGS, C = 1, a maximum of 2,000 iterations, and tolerance 1e-6. The DINOv2 encoder remained frozen in evaluation mode. Photo-residual heads minimized multiclass cross-entropy with an L2 coefficient of 0.001, using zero-initialized, no-intercept weights and deterministic full-batch gradient descent in float64. Training used 400 optimization steps at a constant learning rate of 0.1, without a learning-rate schedule or stochastic data augmentation. These are optimization steps rather than a minibatch epoch count; the effective training batch contained the field-specific eligible training cohort.

The internal research experiment fitted residual and photo-only comparator heads for all 14 fields. The submitted model instead contains one full-training resource with 11 photo-residual heads; its three diagnostic fields retain G0 predictions. The full-training residual resources were fitted with cross-fitted geometry probabilities. Five cross-validation fold models were not averaged or voted at deployment.

## Inference and post-processing

Inference computes the mesh descriptor and, when usable photographs are available, the frozen DINOv2 photo-set feature. The predicted structured findings are used to rerank the five nearest mesh neighbors. A donor replaces the original nearest neighbor only when it reduces the number of field mismatches by at least one and has no more than six mismatches. Otherwise, the original nearest-neighbor report is retained as the donor.

Eligible sentences are replaced using exact structured-finding matches from the training-reference sentence pool. Replacements must preserve sentence count and pass parser checks for both the requested finding changes and unaffected fields. Unsupported, ambiguous, or unsafe replacements are skipped. The final tooth-finding gate removes localized tooth-level crossbite scope. For predefined pathology families, a support threshold of 0.80 among the five mesh neighbors controls retention; retained details are converted to generic wording where applicable. This threshold measures agreement among retrieved reports, not clinical confidence derived from lesion detection.

The pipeline uses deterministic retrieval, category selection, and rule-based text editing. Prompt templates, token decoding temperature, beam width, and top-p sampling are not applicable. Missing or unusable photos use geometry-only predictions. A missing jaw uses the predefined retrieval fallback. Prediction exceptions or a 240-second elapsed prediction deadline return the same predefined nonempty fallback report; resource-loading and input/output errors outside that prediction wrapper still fail. The output is a report in the challenge-required JSON format.

## Did you use an ensemble?

No. Deployment uses one multimodal inference system with field-specific geometry and photo-residual heads. There is no voting or averaging of multiple independently trained full models, and the five cross-validation models are not deployed as an ensemble. The top-five training reports form a retrieval shortlist rather than a five-model ensemble.

## If yes, describe the ensemble

Not applicable.

## Training hardware

The implemented feature extraction and classifier training were performed on a local CPU-only environment, with single-threaded BLAS/OMP settings and a frozen CPU DINOv2 encoder. No GPU was used for this model training. The archived training evidence does not record a reliable CPU model or physical RAM specification, so those details remain unspecified.

For deployment, the platform configuration was No GPU and 16 GB RAM. Separately, the complete 772-input local attempt05 replay used a single Linux/amd64 container with 2 CPUs and a 7 GiB RAM limit, and averaged 15.469 seconds per input. The local replay configuration and timing describe inference, not training hardware or training duration.

## Total training time

The total end-to-end training time was not recorded. Two observed component timings are available: one frozen-DINO CPU feature-cache extraction took approximately 21 minutes, and the corrected five-fold OOF residual/photo-only training runner took 68.62 seconds for 140 fitted heads across 14 fields. The latter includes runner validation and result publication. Neither timing includes the complete geometry-model construction, all repeated verification runs, scoring, and final-resource fitting, so they should not be added and reported as a measured total training duration.

## Number of trainable parameters

The submitted inference model uses 67,902 fitted coefficients and biases: 64,512 photo-residual weights and 3,390 geometry-classifier coefficients and intercepts. The DINOv2 encoder contains 22,056,576 frozen parameters. Scaling statistics, frequency tables, retrieval descriptors, and report/sentence tables are not counted as trainable model coefficients.

For comparison, the earlier 14-field research configuration contained 75,264 photo-head weights per model; that is not the deployed 11-field count. The model archive also retains a legacy V-C classifier with 3,390 coefficients and biases. The submitted loader checks that file's hash but does not use the classifier for prediction. Counting all stored classifier/residual coefficients would therefore give 71,292, while 67,902 is the count actively used by inference.

## External data

The task-specific classifiers, retrieval index, and sentence pool were constructed from the official Bite2Text training release. No additional dental training dataset is identified in the implemented submission path. However, the system uses publicly pretrained DINOv2 weights, which incorporate external pretraining. Accordingly, the method should not be described as using no external pretraining or no external data of any kind.

## Pretrained models

The submitted inference model uses the public `facebook/dinov2-small` DINOv2 ViT-S/14 checkpoint at revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`. The encoder is frozen. No LLM or generative VLM decoder is used in the submitted inference path. The checkpoint's upstream license and redistribution requirements require separate verification before a public release.

## Code, weights, license, and reproducibility status

Public source and active-parameter download links are listed in the repository README. The original project has no applied own-code or own-coefficient licence. A complete final-attempt05 training reproduction has not been verified; the survey reproduction-README answer remains Not yet.

The existing model archive contains training-reference report text, a training-derived sentence pool, and a retrieval descriptor index in addition to classifier and encoder weights. Public release of the entire archive would therefore release data-derived text and index resources as well as weights. Dataset redistribution terms have not been verified. This release separates source code, trained coefficients and upstream checkpoint references. Rebuilding the absent data-dependent resources from an authorized copy of the training data remains outstanding. No license choice or redistribution permission is inferred here.

## Provenance notes

All source references below are relative to the original project root. They identify code or existing evidence; the original project was not modified while preparing this document.

| Claim | Source |
|---|---|
| Final attempt05 wraps the V2 loader with fact reranking, tooth gate, and prediction timeout | `submission/bitevlm_lite_v3_attempt05/__main__.py:17-33`, `submission/bitevlm_lite_v3_attempt05/__main__.py:132-139` |
| V2 resource inventory and fixed model/checkpoint identities | `submission/bitevlm_lite_v2/inference.py:21-58` |
| Deterministic 53-D per-scan and 112-D paired descriptor | `submission/mesh_retrieval_v1/mesh_retrieval_runtime.py:182-250`; deployed mesh manifest `experiments/bitevlm_lite_submission_v2/model_a/mesh_retrieval_v1/manifest.json` records 20,000 triangles per scan |
| Photo preprocessing, fixed 16-page physical batches, CLS/patch feature construction | `vlm_lite/photo_encoder_runtime_v1.py:44-53`, `vlm_lite/photo_encoder_runtime_v1.py:382-415` |
| CPU frozen encoder and page/set feature construction | `vlm_lite/photo_feature_cache.py:1263-1315` |
| Eleven photo fields; three geometry-only diagnostic fields | `vlm_lite/production_runtime_v1.py:43-58`, `vlm_lite/production_runtime_v1.py:176-200` |
| Geometry classifier optimizer | `baselines/e0_field_logistic_oof.py:44-52`, `baselines/e0_field_logistic_oof.py:237-244` |
| Residual loss gradient, scaling, initialization, full-batch optimizer | `vlm_lite/photo_residual.py:424-474` |
| Runner enforces and passes 400 steps, LR 0.1, L2 0.001 | `vlm_lite/p1_runner.py:436-472`, `vlm_lite/p1_runner.py:527-541` |
| Strict nested geometry cross-fitting | `vlm_lite/nested_g0.py:367-382` |
| Patient identity unavailable; duplicate components do not prove patient independence | `VALIDATION_STATUS.md:104-107` |
| Full-training eleven-head resource and fixed optimizer | `vlm_lite/fulltrain_resource.py:72-79`, `vlm_lite/fulltrain_resource.py:694-721` |
| Top-five finding-aware reranking and fixed switch thresholds | `vlm_lite/fact_rerank_runtime_v1.py:90-119` |
| Parser-checked sentence replacement | `baselines/vc_sentence_surgery_runtime.py:423-510` |
| Tooth-finding support threshold and gate behavior | `vlm_lite/tooth_finding_gate_v1.py:17-18`, `vlm_lite/tooth_finding_gate_v1.py:226-235`, `vlm_lite/tooth_finding_gate_v1.py:348-411` |
| Per-case deadline and fallback boundary | `submission/bitevlm_lite_v3_attempt05/__main__.py:30-31`, `PROJECT_LOG.md:14119`, `PROJECT_LOG.md:14135` |
| Approximately 21-minute CPU feature extraction | `PROJECT_LOG.md:8294-8296` |
| 68.62-second corrected OOF training runner, 140 fitted heads | `PROJECT_LOG.md:8324-8326` |
| Local 772-input inference replay, 15.469 seconds/input, matching 2-CPU/7-GiB configuration | `PROJECT_LOG.md:14916-14920` |
| Platform No GPU / 16 GB configuration | `PROJECT_LOG.md:14994-14996` |
| Active learned coefficient counts | Read-only arithmetic on `experiments/bitevlm_lite_submission_v2/model_a/bitevlm_lite_v2/g0_model.json` (`coef`: 3,360; `intercept`: 30) and `photo_residual.json` (`photo_trainable_total`: 64,512), checked against the SHA bindings in `submission/bitevlm_lite_v2/inference.py:38-43` |
| Legacy V-C classifier is hash-checked but not used for inference | `vlm_lite/production_runtime_v1.py:125-155`, `vlm_lite/production_runtime_v1.py:490-511` |
| Model resource contains training reports and report-derived sentences | `submission/mesh_retrieval_v1/mesh_retrieval_runtime.py:307-338`, `baselines/vc_sentence_surgery_runtime.py:210-262` |
| Existing README limitations | `submission/bitevlm_lite_v2/README.md:18-36`, `submission/bitevlm_lite_v3/README.md:1-10`; root LICENSE/README absence checked directly |

## Outstanding information

The complete training wall time, historical training CPU model/RAM, public release URLs, project license, upstream checkpoint redistribution requirements, dataset redistribution terms, and a complete public reproduction guide remain unverified or unavailable. These gaps do not change the implemented architecture described above, but they should not be filled with inferred values.
