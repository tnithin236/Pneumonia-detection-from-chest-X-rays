# 🫁 Pneumonia Detection from Chest X-rays

Fine-tuned a YOLO classification CNN (transfer learning) to detect pneumonia from chest X-rays, optimizing for **recall** and adding **Grad-CAM** heatmaps for explainability. Includes a Streamlit app for interactive testing.

> ⚠️ Educational/portfolio project. Not a medical device and not for clinical use.

## Features
- Transfer learning with `yolov8x-cls` on the Kaggle Chest X-Ray Pneumonia dataset
- Proper train/val/test split (the dataset's 16-image val folder is replaced by a re-split)
- Recall-optimized decision threshold chosen on validation data
- Grad-CAM heatmaps showing which lung regions drive the prediction
- Streamlit UI with an adjustable threshold and multi-image upload

## Project structure
```
pneumonia-detector/
├── app.py                              # Streamlit app
├── models/best.pt                      # trained weights (you add this; git-ignored)
├── notebooks/train_pneumonia_yolo.ipynb  # Colab training notebook
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```
Place your trained weights at `models/best.pt`, then run:
```bash
streamlit run app.py
```
Open http://localhost:8501 and upload a chest X-ray (JPG/PNG).

## Training
Open `notebooks/train_pneumonia_yolo.ipynb` in Google Colab (T4 GPU), add your Kaggle token as the Colab secret `KAGGLE_API_TOKEN`, and run the cells. The notebook trains the model, picks a recall-targeted threshold, generates Grad-CAM examples, and downloads `best.pt`.

## Results
Fill in from your notebook output:

| Metric (test set) | Value |
|---|---|
| Recall (sensitivity) | _ |
| Precision | _ |
| ROC-AUC | _ |
| False negatives | _ |
| Threshold used | _ |

Add `gradcam.png` here: `![Grad-CAM](gradcam.png)`

## Limitations
- The Kaggle test set comes from a slightly different distribution than train, so precision often drops there.
- Pediatric chest X-rays only; results will not generalize to adults or other scanners without more data.
- Grad-CAM shows model attention, not clinical proof. Check that heatmaps highlight lung fields rather than corners or text markers.

## Model weights
`best.pt` for `yolov8x-cls` is around 100+ MB, so it is git-ignored. Share it via GitHub Releases or Git LFS.

## Tech stack
PyTorch, Ultralytics YOLO, pytorch-grad-cam, Streamlit, scikit-learn.
