# Orange Condition Classification

**Greenpod Labs — Assignment 2**
**Labels:** blackspot, canker, fresh, grenning
**Status:** orange dataset added; measured results pending notebook execution.

## 1. Objective

Build and compare image classifiers for four orange image classes. The supplied [Google Drive dataset](https://drive.google.com/file/d/1NMJ0FOxcWH1mwFFeERuY6fgEY-MVgeET/view) is an orange dataset with the labels blackspot, canker, fresh, and grenning. The archive spelling `grenning` is retained to match its folder name. No model weights were provided, so no numerical performance claim or fabricated example outcome is included here.

## 2. Dataset understanding and preparation

The archive is kept in its original split layout under `data/raw/train/<class>/` and `data/raw/test/<class>/`. It contains 991 training images and 99 test images. Notebook 01 checks training-image readability and dimensions, charts class balance, and displays examples; the test counts are recorded below. Dimension outliers are flagged by an interquartile-range rule for human review; this is not an automatic deletion rule. The visual audit should record label ambiguity, duplicate/near-duplicate views, lighting, backgrounds, and acquisition-session differences.

The supplied test split is kept separate. A stratified 15% validation subset is drawn from the training split with random seed 42 and reused for model selection in both workflows. The test partition is reserved for final evaluation. If the dataset contains multiple images of the same fruit or capture session, a group-based split should be considered to reduce leakage.

| Class | Training images |
|---|---:|
| Blackspot | 184 |
| Canker | 179 |
| Fresh | 281 |
| Grenning | 347 |
| Total training | 991 |

Test counts are blackspot 22, canker 22, fresh 33, and grenning 22 (99 total). Dataset quality findings: **[record unreadable files, outlier review, label issues, and possible sampling bias after running notebook 01].**

## 3. Models and selection rationale

### Approach 1 — classical features

RGB channel means and standard deviations summarize global color; HSV histograms describe hue/saturation/value distributions; a uniform Local Binary Pattern histogram captures local texture. The notebook compares an RBF Support Vector Machine with class weighting and a class-weighted Random Forest. These models train quickly, need relatively little data, and allow straightforward feature-level reasoning. Their limitations include sensitivity to illumination and background, and inability to learn spatial/semantic features as flexibly as a CNN.

### Approach 2 — MobileNetV2 transfer learning

MobileNetV2 is initialized with ImageNet weights and a new four-class linear head. The baseline freezes the convolutional feature extractor and trains the head, reducing compute needs and overfitting risk for a small dataset. It may underfit if the domain differs substantially from ImageNet; fine-tuning later blocks is a follow-up to validate, not an assumed improvement. Inference time should be measured on the intended deployment device.

Model selection uses validation macro F1, with validation loss for selecting the deep-learning checkpoint. The held-out test set is only used after this choice. The better test macro F1 can describe this test sample, but does not by itself establish production readiness.

## 4. Evaluation metrics and results

Macro precision, recall, and F1 average per-class scores equally. Macro F1 is the primary comparison when class counts differ, because a frequent class cannot dominate the average. Per-class recall helps identify classes the model misses. Weighted F1 describes performance weighted by sample frequency. Accuracy is an intuitive overall rate but can conceal failure on a rare class. The confusion matrix identifies which labels are confused. Include support counts beside all class-level metrics.

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Best classical validation model | [run notebook 04] | [run notebook 04] | [run notebook 04] | [run notebook 04] | [run notebook 04] |
| MobileNetV2 | [run notebook 04] | [run notebook 04] | [run notebook 04] | [run notebook 04] | [run notebook 04] |

**Test set size and class supports:** [populate from notebook 04].
**Confusion matrices and per-class precision/recall/F1:** [copy notebook 04 output].
**Inference latency:** [report device, batch size, and measured time from notebook 04].

No approach can be declared best until those runs are completed. Select based on macro F1 and per-class recall together with latency and model size, then check errors and limitations.

## 5. Example run

After training, run `python main_demo.py path/to/orange.jpg`. The script prints the predicted class and each class's softmax score. For the submission, replace this section with the actual image filename (or a suitably anonymized identifier), predicted label, score vector, and a brief human review. Softmax scores are not calibrated probabilities and should not be presented as guaranteed confidence.

**Example output:** [run `main_demo.py` on a held-out example and paste the output here].

## 6. Required reflection questions

1. **In production, how does the model perform?** This project currently has no measured production result. Notebook 04 provides an internal held-out estimate once the dataset is run. Production performance must be measured on images from target orchards, cameras, and lighting conditions, with group-separated evaluation. Monitor per-class recall, abstentions/low-confidence cases, latency, and drift after deployment.
2. **What are the drawbacks?** Orange surface conditions can look similar; appearance is affected by lighting, camera, and background. Limited or imbalanced training data can produce weak minority-class recall. Related images across splits may inflate test scores. The frozen MobileNetV2 baseline may underfit, and its softmax scores are not calibrated. External-domain robustness is unproven.
3. **What would you improve with more time?** Obtain more diverse labeled examples and expert-reviewed labels; check for fruit/session overlap; repeat training over seeds; test stronger augmentation and partial backbone fine-tuning using validation-only choices; calibrate scores; evaluate external data; optimize and benchmark on deployment hardware.
4. **Where was the most time spent?** This answer should reflect the candidate's actual work log. A defensible project emphasis is dataset validation and leakage-safe evaluation, since data quality and sampling design determine whether the model comparison is meaningful. Replace this suggested wording with the actual time allocation.

## 7. Reproducibility

Install dependencies with `python -m pip install -r requirements.txt`, extract the archive so its `train/` and `test/` folders are under `data/raw/`, and run notebooks 01, 02, 03, and 04 in order. The notebooks write test features and model checkpoints under ignored `data/processed/` and `models/` paths. See the repository README for the inference command. Dataset files and trained weights are intentionally not included in version control.
