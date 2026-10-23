# Diabetic Retinopathy Detection

A Streamlit web app that predicts the stage of diabetic retinopathy from a retinal
fundus image: **No_DR, Mild, Moderate, Severe, Proliferate_DR**.

## Run the app locally

```bash
git lfs install
git lfs pull                      # downloads archive/my_saved_model.h5 (~230 MB)
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install tensorflow -r requirements.txt
streamlit run app3.py
```

Then open http://localhost:8501, upload an image and click **Predict**.

## Run with Docker

```bash
docker build -t drd .
docker run -p 8501:8501 drd
```

The model is stored with Git LFS, so the build must have the real file, not the LFS
pointer. If the app says the model is a Git LFS pointer, run `git lfs pull` first.

## Model

`archive/my_saved_model.h5` is a CNN trained with `retrain_better_model.py`
(224×224 RGB input scaled to [0, 1]). On the leak-free validation split from
`data_split.py` it scores about 42% accuracy (No_DR precision 0.92; the disease
stages are much weaker). It was trained before that split existed, so even this
figure may be optimistic. Treat predictions as a demo, not a diagnosis.

## Training scripts

| Script | Purpose |
|---|---|
| `retrain_better_model.py` | CNN used by the app. Env overrides: `EPOCHS`, `RESUME_FROM`, `LR`, `BEST_VAL_ACC` |
| `train_model.py` | Simpler CNN baseline |
| `train_transfer_efficientnet.py` | EfficientNetB0 transfer learning (input also [0, 1]) |
| `debug_model.py` | Prints predictions for one image per class |
| `data_split.py` | Train/validation split that keeps duplicated images out of validation |

Training data goes in `archive/colored_images/<class>/` (not tracked in git).

Dataset preparation and synthetic data: `reorganize_data.py`, `balance_dataset.py`,
`fine_tune_all_classes.sh` + `train_dreambooth_lora.py` (LoRA per class),
`generate_synthetic_images.py`, `merge_synthetic_into_dataset.py`. The synthetic
data scripts need `pip install diffusers[torch] transformers accelerate peft` and
a GPU in practice.
