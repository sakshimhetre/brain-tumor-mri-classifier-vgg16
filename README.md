# Brain Tumor MRI Classifier — End to End

Classifies brain MRI images into 4 classes: **glioma, meningioma, pituitary, notumor**.

⚠️ Educational/research project only — not a medical diagnostic tool.

## Files
- `train_brain_tumor_model.ipynb` — Jupyter notebook: downloads data, trains a VGG16
  transfer-learning model, evaluates it, and saves `brain_tumor_model.keras` +
  `class_names.json`.
- `app.py` — Streamlit app that loads the saved model and classifies uploaded MRI images.
- `requirements.txt` — dependencies for the Streamlit app.

## 1. Train the model
1. Open `train_brain_tumor_model.ipynb` in Jupyter, JupyterLab, Google Colab, or Kaggle Notebooks.
2. Get a Kaggle API token (`kaggle.json`) from https://www.kaggle.com/settings and set it up
   as instructed in the notebook's data-download cell.
3. Run all cells top to bottom. This will:
   - Download the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
   - Train + fine-tune a VGG16-based classifier
   - Evaluate on the held-out test set (classification report + confusion matrix)
   - Save `brain_tumor_model.keras` and `class_names.json`

## 2. Run the app locally
```bash
pip install -r requirements.txt
# copy brain_tumor_model.keras and class_names.json into this folder
streamlit run app.py
```

## 3. Deploy to the cloud
### Option A — Streamlit Community Cloud (easiest)
1. Push this folder (including the trained `.keras` model) to a public GitHub repo.
   - If the model file is > 100MB, use [Git LFS](https://git-lfs.com/).
2. Go to https://share.streamlit.io, sign in with GitHub, click "New app".
3. Select your repo/branch and set the main file to `app.py`. Deploy.

### Option B — Hugging Face Spaces
1. Create a new Space, SDK = Streamlit.
2. Upload `app.py`, `requirements.txt`, `brain_tumor_model.keras`, `class_names.json`.
3. Space builds and serves automatically.

### Option C — Docker + Cloud Run / AWS App Runner / Azure Container Apps
Use for production traffic or when you need more control over compute.
Example `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
Build and push to your cloud's container registry, then deploy as a managed service.
