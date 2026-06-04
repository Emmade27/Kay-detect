from __future__ import annotations

import base64
import io
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from flask import (
    Flask,
    flash,
    g,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from werkzeug.utils import secure_filename

from feature_extractor import (
    allowed_extensions,
    decode_image_bytes,
    explain_prediction,
    extract_features_from_image,
)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
INSTANCE_DIR = BASE_DIR / "instance"
DB_PATH = INSTANCE_DIR / "tomato_quality.db"
MODEL_PATH = MODELS_DIR / "tomato_quality_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = "tomato-quality-demo-secret"
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        g.db = conn
    return g.db


@app.teardown_appcontext
def close_db(exception: Optional[BaseException]) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    db = get_db()
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            image_path TEXT NOT NULL,
            source TEXT NOT NULL,
            multiclass_prediction TEXT NOT NULL,
            binary_prediction TEXT NOT NULL,
            multiclass_confidence REAL NOT NULL,
            binary_confidence REAL NOT NULL,
            explanation TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    db.commit()


def ensure_model_exists() -> None:
    if MODEL_PATH.exists() and METADATA_PATH.exists():
        return
    subprocess.run(
        [sys.executable, str(BASE_DIR / "train_model.py"), "--model-dir", str(MODELS_DIR), "--force-demo"],
        check=True,
        cwd=str(BASE_DIR),
    )


def load_model_bundle() -> Tuple[Optional[Any], Dict[str, Any]]:
    try:
        ensure_model_exists()
        model = joblib.load(MODEL_PATH)
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        return model, metadata
    except Exception:
        return None, {
            "class_names": ["Ripe", "Unripe", "Old", "Damaged"],
            "binary_mapping": {"Ripe": "Good Quality", "Unripe": "Low Quality", "Old": "Low Quality", "Damaged": "Low Quality"},
            "dataset_counts": {},
            "demo_model": True,
            "validation_accuracy": None,
        }


MODEL, METADATA = load_model_bundle()


def is_model_ready() -> bool:
    return MODEL is not None


def refresh_model_bundle() -> None:
    global MODEL, METADATA
    MODEL, METADATA = load_model_bundle()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in set(allowed_extensions())


def save_image_bytes(image_bytes: bytes, filename: str) -> Tuple[str, str]:
    safe_name = secure_filename(filename)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    final_name = f"{timestamp}_{safe_name}"
    path = UPLOAD_DIR / final_name
    path.write_bytes(image_bytes)
    rel_path = f"uploads/{final_name}"
    return final_name, rel_path


def decode_camera_data_url(data_url: str) -> bytes:
    if not data_url or "," not in data_url:
        raise ValueError("Invalid camera image data.")
    header, encoded = data_url.split(",", 1)
    if not header.startswith("data:image/"):
        raise ValueError("Camera data is not an image.")
    return base64.b64decode(encoded)


def predict_from_bytes(image_bytes: bytes) -> Dict[str, Any]:
    if not is_model_ready():
        raise RuntimeError("Model is not loaded.")
    image = decode_image_bytes(image_bytes)
    vector, feature_map = extract_features_from_image(image)
    proba = MODEL.predict_proba([vector])[0]
    class_names = list(MODEL.classes_)
    multiclass_idx = int(proba.argmax())
    multiclass_pred = class_names[multiclass_idx]
    multiclass_confidence = float(proba[multiclass_idx])

    ripe_prob = float(proba[class_names.index("Ripe")]) if "Ripe" in class_names else 0.0
    if multiclass_pred == "Ripe":
        binary_pred = "Good Quality"
        binary_confidence = ripe_prob
    else:
        binary_pred = "Low Quality"
        binary_confidence = 1.0 - ripe_prob

    reasons = explain_prediction(multiclass_pred, binary_pred, feature_map)
    return {
        "multiclass_prediction": multiclass_pred,
        "binary_prediction": binary_pred,
        "multiclass_confidence": round(multiclass_confidence * 100, 2),
        "binary_confidence": round(binary_confidence * 100, 2),
        "explanation": reasons,
        "feature_summary": {
            "Redness index": round(float(feature_map["redness_index"]), 2),
            "Dark spot ratio": round(float(feature_map["dark_spot_ratio"]), 3),
            "Texture contrast": round(float(feature_map["glcm_contrast"]), 2),
            "Circularity": round(float(feature_map["circularity"]), 3),
        },
    }


def insert_prediction(record: Dict[str, Any]) -> int:
    db = get_db()
    cursor = db.execute(
        """
        INSERT INTO predictions (
            image_name, image_path, source, multiclass_prediction, binary_prediction,
            multiclass_confidence, binary_confidence, explanation, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            record["image_name"],
            record["image_path"],
            record["source"],
            record["multiclass_prediction"],
            record["binary_prediction"],
            record["multiclass_confidence"],
            record["binary_confidence"],
            json.dumps(record["explanation"]),
            record["created_at"],
        ),
    )
    db.commit()
    return int(cursor.lastrowid)


def get_prediction(record_id: int) -> Optional[sqlite3.Row]:
    db = get_db()
    return db.execute("SELECT * FROM predictions WHERE id = ?", (record_id,)).fetchone()


def query_predictions(limit: Optional[int] = None) -> List[sqlite3.Row]:
    db = get_db()
    query = "SELECT * FROM predictions ORDER BY id DESC"
    if limit:
        return db.execute(query + " LIMIT ?", (limit,)).fetchall()
    return db.execute(query).fetchall()


def stats_payload() -> Dict[str, Any]:
    db = get_db()
    total = db.execute("SELECT COUNT(*) AS c FROM predictions").fetchone()["c"]
    avg_conf = db.execute("SELECT AVG(binary_confidence) AS avg_conf FROM predictions").fetchone()["avg_conf"]
    by_binary = db.execute(
        "SELECT binary_prediction, COUNT(*) AS c FROM predictions GROUP BY binary_prediction"
    ).fetchall()
    by_multiclass = db.execute(
        "SELECT multiclass_prediction, COUNT(*) AS c FROM predictions GROUP BY multiclass_prediction"
    ).fetchall()
    return {
        "total_predictions": total,
        "average_confidence": round(float(avg_conf or 0.0), 2),
        "by_binary": {row["binary_prediction"]: row["c"] for row in by_binary},
        "by_multiclass": {row["multiclass_prediction"]: row["c"] for row in by_multiclass},
        "dataset_counts": METADATA.get("dataset_counts", {}),
        "demo_model": METADATA.get("demo_model", True),
        "validation_accuracy": METADATA.get("validation_accuracy"),
        "dataset_source": METADATA.get("dataset_source", "not configured"),
    }


def dataframe_for_predictions() -> pd.DataFrame:
    rows = query_predictions()
    payload = []
    for row in rows:
        payload.append({
            "ID": row["id"],
            "Image Name": row["image_name"],
            "Source": row["source"],
            "Multiclass": row["multiclass_prediction"],
            "Binary": row["binary_prediction"],
            "Multiclass Confidence": row["multiclass_confidence"],
            "Binary Confidence": row["binary_confidence"],
            "Created At": row["created_at"],
        })
    return pd.DataFrame(payload)


@app.before_request
def before_request() -> None:
    init_db()


@app.route("/")
def home():
    return render_template("home.html", model_ready=is_model_ready(), metadata=METADATA)


@app.route("/predict-page")
def predict_page():
    return render_template("prediction.html", model_ready=is_model_ready(), metadata=METADATA)


@app.route("/predict", methods=["POST"])
def predict():
    if not is_model_ready():
        flash("The model is not loaded. Train or reload the model before prediction.", "danger")
        return redirect(url_for("predict_page"))

    source = "Upload"
    image_bytes = None
    image_name = None

    camera_data = request.form.get("camera_data", "").strip()
    upload = request.files.get("image")

    try:
        if camera_data:
            source = "Camera"
            image_bytes = decode_camera_data_url(camera_data)
            image_name = "camera_capture.png"
        elif upload and upload.filename:
            if not allowed_file(upload.filename):
                flash("Invalid file type. Please upload JPG, PNG, JPEG, BMP, or WEBP.", "warning")
                return redirect(url_for("predict_page"))
            image_bytes = upload.read()
            image_name = upload.filename
        else:
            flash("No image selected. Upload a file or capture a photo in the app.", "warning")
            return redirect(url_for("predict_page"))

        saved_name, relative_path = save_image_bytes(image_bytes, image_name)
        prediction = predict_from_bytes(image_bytes)
        record_id = insert_prediction(
            {
                "image_name": saved_name,
                "image_path": relative_path,
                "source": source,
                "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                **prediction,
            }
        )
        return redirect(url_for("result", record_id=record_id))
    except ValueError as exc:
        flash(str(exc), "danger")
        return redirect(url_for("predict_page"))
    except Exception as exc:
        flash(f"Prediction failed: {exc}", "danger")
        return redirect(url_for("predict_page"))


@app.route("/result/<int:record_id>")
def result(record_id: int):
    row = get_prediction(record_id)
    if row is None:
        return render_template("error.html", title="Prediction not found", message="The requested prediction result could not be found."), 404
    explanation = json.loads(row["explanation"])
    return render_template("result.html", row=row, explanation=explanation)


@app.route("/history")
def history():
    return render_template("history.html", predictions=query_predictions())


@app.route("/admin")
def admin():
    return render_template("admin.html", stats=stats_payload(), predictions=query_predictions(limit=10), model_ready=is_model_ready())


@app.route("/admin/retrain", methods=["POST"])
def retrain_model():
    dataset_path = request.form.get("dataset_path", "").strip()
    if not dataset_path:
        flash("Provide the Kaggle dataset folder path containing Ripe, Unripe, Old, and Damaged folders.", "warning")
        return redirect(url_for("admin"))
    try:
        subprocess.run(
            [sys.executable, str(BASE_DIR / "train_model.py"), "--dataset", dataset_path, "--model-dir", str(MODELS_DIR)],
            check=True,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
        )
        refresh_model_bundle()
        flash("Model retraining completed successfully.", "success")
    except subprocess.CalledProcessError as exc:
        message = exc.stderr[-400:] if exc.stderr else str(exc)
        flash(f"Retraining failed: {message}", "danger")
    return redirect(url_for("admin"))


@app.route("/admin/export/csv")
def export_csv():
    df = dataframe_for_predictions()
    csv_data = df.to_csv(index=False)
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=tomato_predictions.csv"
    response.headers["Content-Type"] = "text/csv"
    return response


@app.route("/admin/export/pdf")
def export_pdf():
    rows = query_predictions(limit=30)
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, "Tomato Quality Detection Report")
    y -= 30
    pdf.setFont("Helvetica", 9)
    for row in rows:
        line = f"#{row['id']} | {row['image_name']} | {row['binary_prediction']} | {row['binary_confidence']}% | {row['created_at']}"
        pdf.drawString(40, y, line[:120])
        y -= 16
        if y < 60:
            pdf.showPage()
            pdf.setFont("Helvetica", 9)
            y = height - 40
    pdf.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="tomato_predictions.pdf", mimetype="application/pdf")


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": is_model_ready(),
        "database": DB_PATH.exists(),
        "supported_formats": [ext.upper() for ext in allowed_extensions()],
    })


@app.errorhandler(404)
def not_found(error):
    return render_template("error.html", title="Page not found", message="The page you requested does not exist."), 404


@app.errorhandler(413)
def too_large(error):
    return render_template("error.html", title="File too large", message="The uploaded file exceeds the 8 MB limit."), 413


@app.errorhandler(500)
def server_error(error):
    return render_template("error.html", title="Server error", message="An unexpected error occurred while processing your request."), 500


if __name__ == "__main__":
    ensure_model_exists()
    refresh_model_bundle()
    app.run(debug=True)
