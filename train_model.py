import os
import numpy as np
import pandas as pd
import joblib

print("=" * 50)
print("Alzheimer Risk Assessment AI")
print("=" * 50)

# ==========================================
# ตำแหน่งไฟล์
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "alzheimers_disease_data.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "alzheimer_model.pkl"
)

# ==========================================
# โหลด Dataset
# ==========================================

df = pd.read_csv(DATASET_PATH)

# ==========================================
# Features ที่ใช้ในการประเมิน
# ==========================================

features = [
    "Age",
    "Gender",
    "MemoryComplaints",
    "Confusion",
    "FamilyHistoryAlzheimers",
    "Disorientation",
    "DifficultyCompletingTasks"
]

target = "Diagnosis"

# ตรวจสอบว่าคอลัมน์ครบหรือไม่
missing_columns = [
    column for column in features + [target]
    if column not in df.columns
]

if missing_columns:
    print("ไม่พบคอลัมน์:")
    for column in missing_columns:
        print("-", column)
    raise ValueError("Dataset มีคอลัมน์ไม่ครบ")

X = df[features].values.astype(float)
y = df[target].values.astype(float)

# ==========================================
# Shuffle
# ==========================================

np.random.seed(42)

indices = np.arange(len(X))
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

# ==========================================
# Train / Test Split
# ==========================================

split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# ==========================================
# Standardization
# ==========================================

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

std[std == 0] = 1

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

# ==========================================
# เพิ่ม Bias
# ==========================================

X_train = np.c_[
    np.ones(len(X_train)),
    X_train
]

X_test = np.c_[
    np.ones(len(X_test)),
    X_test
]

# ==========================================
# Logistic Regression
# ==========================================

weights = np.zeros(
    X_train.shape[1]
)

learning_rate = 0.01
epochs = 3000

for epoch in range(epochs):

    z = X_train @ weights

    z = np.clip(
        z,
        -500,
        500
    )

    predictions = (
        1 /
        (
            1 + np.exp(-z)
        )
    )

    gradient = (
        X_train.T @
        (predictions - y_train)
        / len(X_train)
    )

    weights -= (
        learning_rate *
        gradient
    )

# ==========================================
# ทดสอบโมเดล
# ==========================================

z = X_test @ weights

z = np.clip(
    z,
    -500,
    500
)

probabilities = (
    1 /
    (
        1 + np.exp(-z)
    )
)

predictions = (
    probabilities >= 0.5
).astype(int)

accuracy = np.mean(
    predictions == y_test
)

correct = np.sum(
    predictions == y_test
)

wrong = np.sum(
    predictions != y_test
)

# ==========================================
# แสดงผล
# ==========================================

print()
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")

print()
print(f"จำนวนข้อมูลทั้งหมด: {len(X)}")
print(f"ข้อมูลสำหรับ Train: {len(X_train)}")
print(f"ข้อมูลสำหรับ Test: {len(X_test)}")

print()
print("Features ที่ใช้:")

for feature in features:
    print(f"- {feature}")

print()
print("ผลการทำนาย:")
print(f"ถูก: {correct}")
print(f"ผิด: {wrong}")

# ==========================================
# บันทึกโมเดล
# ==========================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

model = {
    "weights": weights,
    "mean": mean,
    "std": std,
    "features": features
}

joblib.dump(
    model,
    MODEL_PATH
)

print()
print("บันทึกโมเดลเรียบร้อยแล้ว")
print(f"ไฟล์: {MODEL_PATH}")