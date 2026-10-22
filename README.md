# Orange fruit condition classification

Computer vision project comparing handcrafted color/texture features (RBF SVM and Random Forest) with a MobileNetV2 transfer-learning classifier across **blackspot, canker, fresh, and grenning** orange image classes.

## Data setup

The provided [Google Drive archive](https://drive.google.com/file/d/1NMJ0FOxcWH1mwFFeERuY6fgEY-MVgeET/view) is named `Orange Dataset.zip`. Keep its existing training and test split when extracting:

```text
data/raw/train/blackspot/
data/raw/train/canker/
data/raw/train/fresh/
data/raw/train/grenning/
data/raw/test/blackspot/
data/raw/test/canker/
data/raw/test/fresh/
data/raw/test/grenning/
```

The archive contains 991 training images and 99 test images. Notebook 01 audits the training split. Notebooks 02 and 03 create a stratified validation split from the training folder and reserve the supplied test folder for final evaluation. Do not move test images into training. Keep the original images out of version control.

## Setup and execution

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter notebook
```

Run notebooks 01 through 04 in order. They audit the data, train and compare the classical models and MobileNetV2, then evaluate the selected models on the supplied test set. The validation split is stratified and seeded; do not tune on the test set.

After notebook 03 saves `models/mobilenetv2_best.pth`, run an example inference on an orange image:

```bash
python main_demo.py path/to/orange.jpg
```

## Results

Notebook 04 computes accuracy, macro and weighted precision/recall/F1, per-class recall, confusion matrix, and inference latency from held-out predictions. Update the report with measured results after running the notebooks; do not infer performance from the training data.

## Approach and evaluation

The classes are the labels supplied with the orange archive: blackspot, canker, fresh, and grenning. The archive's label spelling `grenning` is retained to match its folders. The classical baseline uses RGB statistics, HSV histograms, and a uniform Local Binary Pattern histogram. It is interpretable and inexpensive, but may be sensitive to lighting, background, and camera differences. MobileNetV2 offers a compact pretrained feature extractor and is a useful transfer-learning baseline. Its CNN head is trained first with the pretrained backbone frozen; further fine-tuning should be justified using validation results and compute constraints.

Macro F1 gives each class equal weight, so strong performance on common classes cannot hide poor results on a rare class. Per-class recall measures the fraction of each true class found, and the confusion matrix shows which labels are confused. Accuracy is included as an intuitive overall summary, not used alone. The supplied split is retained; check for related-image leakage if the collection contains multiple views of the same fruit.

See [report/Assignment2_Avocado_Report.pdf](report/Assignment2_Avocado_Report.pdf) and the editable [report/Assignment2_Avocado_Report.md](report/Assignment2_Avocado_Report.md).
