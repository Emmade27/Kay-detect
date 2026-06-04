# Tomato Quality Detection WebApp Using Feature Extraction and Machine Learning

A complete Flask web application for tomato quality detection using image preprocessing, feature extraction, and a trained machine learning classifier.

## Project modules
- Home page
- Prediction page with image upload and in-app camera capture
- Result page with confidence score and explanation
- Prediction history page
- Admin dashboard with exports and retraining
- Health check route

## Kaggle dataset structure expected by the training script
The training script expects a dataset root folder containing these four subfolders:

```text
TomatoesDataset/
├── Ripe/
├── Unripe/
├── Old/
└── Damaged/
```

The app uses this mapping:
- Good Quality = Ripe
- Low Quality = Unripe, Old, Damaged

## Sections that need external installation before code execution
These parts require external setup on your system:

1. **Python installation**
   - Install Python 3.10 or newer.

2. **Python packages from `requirements.txt`**
   - Flask
   - joblib
   - numpy
   - opencv-python
   - pandas
   - scikit-image
   - scikit-learn
   - reportlab

3. **Kaggle dataset download**
   - You need to download the tomato image dataset and place it in folders named `Ripe`, `Unripe`, `Old`, and `Damaged` if you want to retrain the model with real data.

4. **Browser camera access**
   - The in-app camera feature uses the browser `MediaDevices` API.
   - It works in modern browsers on **localhost** or **HTTPS**.
   - No extra Python package is required for the camera itself, but browser permission is required.

## What does not need external installation
- SQLite does not need separate installation because Python includes `sqlite3` in the standard library.
- File upload support is already handled by Flask and Werkzeug once dependencies are installed.

## Quick start

### 1. Create and activate a virtual environment
```bash
python -m venv .venv
```

Windows PowerShell:
```bash
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:
```bash
.\.venv\Scripts\activate.bat
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

The app will create a **demo model automatically** if no trained model exists yet.

Open your browser at:
```text
http://127.0.0.1:5000
```

## Train using the Kaggle dataset
```bash
python train_model.py --dataset "C:\Users\YourName\Downloads\TomatoesDataset" --model-dir models
```

This will save:
- `models/tomato_quality_model.joblib`
- `models/metadata.json`

## Main routes
- `/` Home page
- `/predict-page` Prediction page
- `/predict` Prediction submission route
- `/result/<id>` Result page
- `/history` Prediction history
- `/admin` Admin dashboard
- `/admin/export/csv` Export CSV
- `/admin/export/pdf` Export PDF
- `/health` Health check

## Notes
- The packaged app includes automatic demo-model generation so the interface is usable before real training.
- For real project evaluation, retrain the model with the Kaggle dataset.
