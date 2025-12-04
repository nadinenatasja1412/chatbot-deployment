import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from chat import get_response as chat_get_response

# --- Konstanta umum ---
MODEL_PATH = Path("model_malnutrisi.pkl")
N_SAMPLES = 20000
RANDOM_STATE = 42

# --- Logika WHO Z-Score Gabungan (-2 SD & +2 SD) untuk Pelabelan (0 - 60 Bulan) ---
# (Dipotong untuk kejelasan: data sama seperti implementasi terdahulu)
BATAS_WHO_ZSCORE = {
    "Laki-laki": {
        "BB_MIN": {
            0: 2.5,
            1: 3.4,
            2: 4.3,
            3: 5.0,
            4: 5.6,
            5: 6.0,
            6: 6.4,
            7: 6.7,
            8: 6.9,
            9: 7.1,
            10: 7.4,
            11: 7.6,
            12: 7.7,
            13: 7.9,
            14: 8.1,
            15: 8.3,
            16: 8.4,
            17: 8.6,
            18: 8.8,
            19: 8.9,
            20: 9.1,
            21: 9.2,
            22: 9.5,
            23: 9.7,
            24: 9.8,
            25: 10.0,
            26: 10.1,
            27: 10.2,
            28: 10.4,
            29: 10.5,
            30: 10.7,
            31: 10.8,
            32: 11.0,
            33: 11.1,
            34: 11.2,
            35: 11.3,
            36: 11.5,
            37: 11.6,
            38: 11.7,
            39: 11.9,
            40: 12.0,
            41: 12.1,
            42: 12.3,
            43: 12.4,
            44: 12.5,
            45: 12.6,
            46: 12.7,
            47: 12.8,
            48: 12.9,
            49: 13.0,
            50: 13.1,
            51: 13.2,
            52: 13.3,
            53: 13.4,
            54: 13.5,
            55: 13.6,
            56: 13.6,
            57: 13.7,
            58: 13.8,
            59: 13.9,
            60: 14.0,
        },
        "BB_MAX": {
            0: 4.8,
            1: 6.6,
            2: 7.9,
            3: 8.9,
            4: 9.7,
            5: 10.3,
            6: 10.7,
            7: 11.0,
            8: 11.4,
            9: 11.7,
            10: 12.0,
            11: 12.3,
            12: 12.5,
            13: 12.8,
            14: 13.0,
            15: 13.3,
            16: 13.5,
            17: 13.8,
            18: 14.0,
            19: 14.3,
            20: 14.5,
            21: 14.8,
            22: 15.0,
            23: 15.3,
            24: 15.5,
            25: 15.7,
            26: 16.0,
            27: 16.2,
            28: 16.4,
            29: 16.6,
            30: 16.9,
            31: 17.1,
            32: 17.3,
            33: 17.5,
            34: 17.8,
            35: 18.0,
            36: 18.2,
            37: 18.4,
            38: 18.6,
            39: 18.8,
            40: 19.1,
            41: 19.3,
            42: 19.5,
            43: 19.7,
            44: 19.9,
            45: 20.1,
            46: 20.3,
            47: 20.5,
            48: 20.7,
            49: 20.9,
            50: 21.1,
            51: 21.3,
            52: 21.5,
            53: 21.7,
            54: 21.8,
            55: 22.0,
            56: 22.2,
            57: 22.4,
            58: 22.5,
            59: 22.7,
            60: 22.9,
        },
        "TB_MIN": {
            0: 46.1,
            1: 50.8,
            2: 54.4,
            3: 57.3,
            4: 59.7,
            5: 61.7,
            6: 63.3,
            7: 64.8,
            8: 66.2,
            9: 67.5,
            10: 68.7,
            11: 69.9,
            12: 71.0,
            13: 72.1,
            14: 73.1,
            15: 74.1,
            16: 75.0,
            17: 76.0,
            18: 76.9,
            19: 77.7,
            20: 78.6,
            21: 79.4,
            22: 80.2,
            23: 81.0,
            24: 81.7,
            25: 82.4,
            26: 83.1,
            27: 83.7,
            28: 84.5,
            29: 85.1,
            30: 85.7,
            31: 86.4,
            32: 86.9,
            33: 87.5,
            34: 88.1,
            35: 88.7,
            36: 89.2,
            37: 89.7,
            38: 90.3,
            39: 90.8,
            40: 91.4,
            41: 91.9,
            42: 92.4,
            43: 92.9,
            44: 93.4,
            45: 93.9,
            46: 94.4,
            47: 94.9,
            48: 95.4,
            49: 95.8,
            50: 96.4,
            51: 96.9,
            52: 97.4,
            53: 97.8,
            54: 98.3,
            55: 98.8,
            56: 99.3,
            57: 99.7,
            58: 100.2,
            59: 100.7,
            60: 101.2,
        },
        "TB_MAX": {
            0: 51.5,
            1: 56.4,
            2: 60.1,
            3: 63.3,
            4: 65.9,
            5: 68.0,
            6: 69.8,
            7: 71.4,
            8: 72.9,
            9: 74.3,
            10: 75.6,
            11: 76.9,
            12: 78.1,
            13: 79.2,
            14: 80.3,
            15: 81.4,
            16: 82.4,
            17: 83.4,
            18: 84.4,
            19: 85.3,
            20: 86.2,
            21: 87.1,
            22: 87.9,
            23: 88.7,
            24: 89.5,
            25: 90.3,
            26: 91.0,
            27: 91.7,
            28: 92.4,
            29: 93.1,
            30: 93.8,
            31: 94.4,
            32: 95.1,
            33: 95.7,
            34: 96.3,
            35: 96.9,
            36: 97.5,
            37: 98.1,
            38: 98.6,
            39: 99.2,
            40: 99.7,
            41: 100.3,
            42: 100.8,
            43: 101.4,
            44: 101.9,
            45: 102.4,
            46: 102.9,
            47: 103.4,
            48: 103.9,
            49: 104.4,
            50: 104.9,
            51: 105.4,
            52: 105.8,
            53: 106.3,
            54: 106.8,
            55: 107.2,
            56: 107.7,
            57: 108.1,
            58: 108.6,
            59: 109.0,
            60: 109.4,
        },
    },
    "Perempuan": {
        "BB_MIN": {
            0: 2.4,
            1: 3.2,
            2: 4.0,
            3: 4.6,
            4: 5.0,
            5: 5.4,
            6: 5.7,
            7: 6.0,
            8: 6.3,
            9: 6.5,
            10: 6.7,
            11: 6.9,
            12: 7.0,
            13: 7.2,
            14: 7.4,
            15: 7.6,
            16: 7.7,
            17: 7.9,
            18: 8.1,
            19: 8.2,
            20: 8.4,
            21: 8.6,
            22: 8.7,
            23: 8.9,
            24: 9.0,
            25: 9.2,
            26: 9.3,
            27: 9.5,
            28: 9.7,
            29: 9.8,
            30: 10.0,
            31: 10.1,
            32: 10.3,
            33: 11.4,
            34: 10.5,
            35: 10.7,
            36: 10.8,
            37: 10.9,
            38: 11.1,
            39: 11.3,
            40: 11.5,
            41: 11.6,
            42: 11.7,
            43: 11.9,
            44: 12.1,
            45: 12.2,
            46: 12.3,
            47: 12.5,
            48: 12.6,
            49: 12.7,
            50: 12.8,
            51: 12.9,
            52: 13.0,
            53: 13.2,
            54: 13.3,
            55: 13.4,
            56: 13.5,
            57: 13.6,
            58: 13.7,
            59: 13.8,
            60: 13.7,
        },
        "BB_MAX": {
            0: 4.5,
            1: 6.2,
            2: 7.4,
            3: 8.3,
            4: 9.0,
            5: 9.5,
            6: 10.0,
            7: 10.3,
            8: 10.6,
            9: 10.9,
            10: 11.2,
            11: 11.5,
            12: 11.8,
            13: 12.0,
            14: 12.3,
            15: 12.5,
            16: 12.8,
            17: 13.0,
            18: 13.3,
            19: 13.5,
            20: 13.8,
            21: 14.0,
            22: 14.3,
            23: 14.5,
            24: 14.7,
            25: 15.0,
            26: 15.2,
            27: 15.4,
            28: 15.6,
            29: 15.8,
            30: 16.1,
            31: 16.3,
            32: 16.5,
            33: 16.7,
            34: 17.0,
            35: 17.2,
            36: 17.4,
            37: 17.6,
            38: 17.8,
            39: 18.1,
            40: 18.3,
            41: 18.5,
            42: 18.7,
            43: 18.9,
            44: 19.1,
            45: 19.3,
            46: 19.5,
            47: 19.7,
            48: 19.9,
            49: 20.1,
            50: 20.3,
            51: 20.5,
            52: 20.7,
            53: 20.8,
            54: 21.0,
            55: 21.2,
            56: 21.4,
            57: 21.6,
            58: 21.7,
            59: 21.9,
            60: 22.1,
        },
        "TB_MIN": {
            0: 45.4,
            1: 49.8,
            2: 53.0,
            3: 55.6,
            4: 57.6,
            5: 59.6,
            6: 61.2,
            7: 62.7,
            8: 64.0,
            9: 65.3,
            10: 66.5,
            11: 67.7,
            12: 68.9,
            13: 70.0,
            14: 71.0,
            15: 72.0,
            16: 73.0,
            17: 74.0,
            18: 74.9,
            19: 75.8,
            20: 76.7,
            21: 77.5,
            22: 78.4,
            23: 79.2,
            24: 80.0,
            25: 80.8,
            26: 81.5,
            27: 82.2,
            28: 82.9,
            29: 83.6,
            30: 84.3,
            31: 84.9,
            32: 85.6,
            33: 86.2,
            34: 86.8,
            35: 87.4,
            36: 88.0,
            37: 88.6,
            38: 89.2,
            39: 89.8,
            40: 90.4,
            41: 91.0,
            42: 91.5,
            43: 92.1,
            44: 92.5,
            45: 93.0,
            46: 93.6,
            47: 94.1,
            48: 94.6,
            49: 95.1,
            50: 95.6,
            51: 96.1,
            52: 96.6,
            53: 97.1,
            54: 97.6,
            55: 98.1,
            56: 98.5,
            57: 99.0,
            58: 99.5,
            59: 99.9,
            60: 100.3,
        },
        "TB_MAX": {
            0: 50.8,
            1: 55.4,
            2: 58.8,
            3: 61.6,
            4: 63.8,
            5: 65.8,
            6: 67.5,
            7: 69.0,
            8: 70.4,
            9: 71.7,
            10: 73.0,
            11: 74.3,
            12: 75.5,
            13: 76.6,
            14: 77.7,
            15: 78.8,
            16: 79.8,
            17: 80.8,
            18: 81.7,
            19: 82.6,
            20: 83.5,
            21: 84.4,
            22: 85.2,
            23: 86.0,
            24: 86.8,
            25: 87.6,
            26: 88.4,
            27: 89.1,
            28: 89.8,
            29: 90.5,
            30: 91.2,
            31: 91.9,
            32: 92.5,
            33: 93.2,
            34: 93.8,
            35: 94.4,
            36: 95.0,
            37: 95.6,
            38: 96.2,
            39: 96.8,
            40: 97.3,
            41: 97.9,
            42: 98.4,
            43: 99.0,
            44: 99.5,
            45: 100.0,
            46: 100.5,
            47: 101.0,
            48: 101.5,
            49: 102.0,
            50: 102.5,
            51: 103.0,
            52: 103.5,
            53: 103.9,
            54: 104.4,
            55: 104.9,
            56: 105.3,
            57: 105.8,
            58: 106.2,
            59: 106.7,
            60: 107.1,
        },
    },
}


def get_antropometri_limits(gender: str, usia_bulan: int) -> Tuple[float, float, float, float]:
    data_gender = BATAS_WHO_ZSCORE[gender]
    bb_min = data_gender["BB_MIN"].get(usia_bulan)
    bb_max = data_gender["BB_MAX"].get(usia_bulan)
    tb_min = data_gender["TB_MIN"].get(usia_bulan)
    tb_max = data_gender["TB_MAX"].get(usia_bulan)
    return bb_min, bb_max, tb_min, tb_max


def label_malnutrisi_komprehensif(row: pd.Series) -> str:
    gejala_count = row["kulit_rambut"] + row["otot_perut"] + row["imunitas"]
    if gejala_count >= 2:
        return "Malnutrisi"

    usia = row["usia_bulan"]
    berat = row["berat_kg"]
    tinggi = row["tinggi_cm"]
    gender = row["jenis_kelamin"]

    berat_min, berat_max, tinggi_min, tinggi_max = get_antropometri_limits(gender, usia)

    if berat_min is None or tinggi_min is None:
        return "Normal"

    if berat < berat_min or tinggi < tinggi_min:
        return "Malnutrisi"
    if berat > berat_max:
        return "Malnutrisi"

    return "Normal"


def generate_training_dataframe() -> pd.DataFrame:
    np.random.seed(RANDOM_STATE)
    data = {
        "usia_bulan": np.random.randint(0, 60, N_SAMPLES),
        "berat_kg": np.random.normal(loc=12.0, scale=4.0, size=N_SAMPLES),
        "tinggi_cm": np.random.normal(loc=85.0, scale=15.0, size=N_SAMPLES),
        "jenis_kelamin": np.random.choice(["Laki-laki", "Perempuan"], N_SAMPLES),
        "kulit_rambut": np.random.choice([0, 1], N_SAMPLES, p=[0.80, 0.20]),
        "otot_perut": np.random.choice([0, 1], N_SAMPLES, p=[0.85, 0.15]),
        "imunitas": np.random.choice([0, 1], N_SAMPLES, p=[0.75, 0.25]),
    }
    df = pd.DataFrame(data)
    df["status_gizi"] = df.apply(label_malnutrisi_komprehensif, axis=1)
    le = LabelEncoder()
    df["jenis_kelamin_encoded"] = le.fit_transform(df["jenis_kelamin"])
    return df


def train_and_save_model() -> RandomForestClassifier:
    print("Training RandomForest model for ChildHealth Monitor...")
    df = generate_training_dataframe()
    X = df[
        [
            "usia_bulan",
            "berat_kg",
            "tinggi_cm",
            "jenis_kelamin_encoded",
            "kulit_rambut",
            "otot_perut",
            "imunitas",
        ]
    ]
    y = df["status_gizi"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to '{MODEL_PATH}' with accuracy {accuracy:.4f}")
    return model


def load_or_train_model() -> RandomForestClassifier:
    if MODEL_PATH.exists():
        try:
            print("Loading existing malnutrition model...")
            return joblib.load(MODEL_PATH)
        except Exception as exc:
            print(f"Failed to load model ({exc}), retraining...")
    return train_and_save_model()


model = load_or_train_model()

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health_check():
    status = "ready" if model is not None else "unavailable"
    return jsonify({"status": status})


@app.route("/predict_gizi", methods=["POST"])
def predict_gizi():
    if model is None:
        return jsonify({"error": "Model tidak tersedia."}), 500

    data = request.get_json(force=True)
    try:
        payload = {
            "usia_bulan": data["usia_bulan"],
            "berat_kg": data["berat_kg"],
            "tinggi_cm": data["tinggi_cm"],
            "jenis_kelamin_encoded": data["jenis_kelamin_encoded"],
            "kulit_rambut": data["kulit_rambut"],
            "otot_perut": data["otot_perut"],
            "imunitas": data["imunitas"],
        }
    except KeyError as missing:
        return jsonify({"error": f"Missing field: {missing}"}), 400

    input_df = pd.DataFrame([payload])
    try:
        prediction = model.predict(input_df)[0]
    except Exception as exc:
        return jsonify({"error": f"Gagal memproses input: {exc}"}), 400

    return jsonify({"status_gizi": prediction, "message": "Prediksi berhasil"})


@app.route("/train", methods=["POST"])
def retrain():
    """
    Endpoint opsional untuk memicu pelatihan ulang model.
    """
    global model
    model = train_and_save_model()
    return jsonify({"message": "Model berhasil dilatih ulang."})


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """
    Endpoint chatbot sederhana berbasis model intent PyTorch.
    Body JSON: {"message": "teks pengguna"}
    """
    data = request.get_json(force=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Field 'message' wajib diisi."}), 400

    try:
        reply = chat_get_response(message)
    except Exception as exc:
        return jsonify({"error": f"Gagal memproses pesan: {exc}"}), 500

    return jsonify({"response": reply})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)