from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# ==========================================
# โหลดโมเดล
# ==========================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "alzheimer_model.pkl"
)

try:

    model = joblib.load(MODEL_PATH)

    weights = model["weights"]
    mean = model["mean"]
    std = model["std"]

except Exception as e:

    weights = None
    mean = None
    std = None

    print("ไม่สามารถโหลดโมเดลได้:", e)


# ==========================================
# ฟังก์ชัน AI
# ==========================================

def predict_risk(
    age,
    gender,
    memory,
    confusion,
    family_history,
    disorientation,
    difficulty_tasks
):

    features = np.array(
        [[
            age,
            gender,
            memory,
            confusion,
            family_history,
            disorientation,
            difficulty_tasks
        ]],
        dtype=float
    )

    # Standardization
    features = (
        features - mean
    ) / std

    # เพิ่ม Bias
    features = np.c_[
        np.ones(len(features)),
        features
    ]

    # Logistic Regression
    z = features @ weights

    z = np.clip(
        z,
        -500,
        500
    )

    probability = (
        1 /
        (
            1 + np.exp(-z)
        )
    )

    return float(
        probability[0]
    )


# ==========================================
# หน้าแรก
# ==========================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ==========================================
# ประเมินความเสี่ยง
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # -----------------------------
        # รับข้อมูล
        # -----------------------------

        age = float(
            request.form.get(
                "age",
                0
            )
        )

        gender_text = request.form.get(
            "gender",
            ""
        )

        memory = int(
            request.form.get(
                "memory",
                0
            )
        )

        confusion = int(
            request.form.get(
                "confusion",
                0
            )
        )

        family_history = int(
            request.form.get(
                "family_history",
                0
            )
        )

        disorientation = int(
            request.form.get(
                "disorientation",
                0
            )
        )

        difficulty_tasks = int(
            request.form.get(
                "difficulty_tasks",
                0
            )
        )

        # -----------------------------
        # ตรวจสอบอายุ
        # -----------------------------

        if age < 1 or age > 120:

            return render_template(
                "index.html",
                error="กรุณากรอกอายุระหว่าง 1-120 ปี"
            )

        # -----------------------------
        # ตรวจสอบเพศ
        # -----------------------------

        if gender_text not in [
            "male",
            "female"
        ]:

            return render_template(
                "index.html",
                error="กรุณาเลือกเพศ"
            )

        # -----------------------------
        # ตรวจสอบคำตอบ
        # -----------------------------

        answers = [
            memory,
            confusion,
            family_history,
            disorientation,
            difficulty_tasks
        ]

        if any(
            answer not in [0, 1]
            for answer in answers
        ):

            return render_template(
                "index.html",
                error="ข้อมูลแบบประเมินไม่ถูกต้อง"
            )

        # -----------------------------
        # แปลงเพศ
        # -----------------------------

        if gender_text == "male":

            gender = 0

        else:

            gender = 1

        # -----------------------------
        # ตรวจสอบโมเดล
        # -----------------------------

        if weights is None:

            return render_template(
                "index.html",
                error="ไม่พบโมเดล AI กรุณาตรวจสอบไฟล์ model/alzheimer_model.pkl"
            )

        # -----------------------------
        # เรียก AI
        # -----------------------------

        probability = predict_risk(
            age,
            gender,
            memory,
            confusion,
            family_history,
            disorientation,
            difficulty_tasks
        )

        # -----------------------------
        # เปอร์เซ็นต์
        # -----------------------------

        risk_percent = round(
            probability * 100,
            2
        )

        # -----------------------------
        # ระดับความเสี่ยง
        # -----------------------------

        if probability < 0.30:

            risk_level = "ความเสี่ยงต่ำ"

        elif probability < 0.60:

            risk_level = "ความเสี่ยงปานกลาง"

        else:

            risk_level = "ความเสี่ยงสูง"

        # -----------------------------
        # คำแนะนำ
        # -----------------------------

        if probability < 0.30:

            recommendation = (
                "ผลการประเมินจากข้อมูลที่กรอก "
                "อยู่ในระดับความเสี่ยงต่ำ "
                "ควรดูแลสุขภาพและติดตามอาการอย่างสม่ำเสมอ"
            )

        elif probability < 0.60:

            recommendation = (
                "ผลการประเมินอยู่ในระดับความเสี่ยงปานกลาง "
                "ควรติดตามการเปลี่ยนแปลงของความจำ "
                "และปรึกษาผู้เชี่ยวชาญหากมีอาการต่อเนื่อง"
            )

        else:

            recommendation = (
                "ผลการประเมินอยู่ในระดับความเสี่ยงสูง "
                "ควรปรึกษาแพทย์หรือบุคลากรทางการแพทย์ "
                "เพื่อประเมินเพิ่มเติม"
            )

        # -----------------------------
        # ส่งผลลัพธ์
        # -----------------------------

        return render_template(

            "result.html",

            result=risk_level,

            risk_percent=risk_percent,

            recommendation=recommendation,

            age=int(age),

            gender=gender_text,

            memory=memory,

            confusion=confusion,

            family_history=family_history,

            disorientation=disorientation,

            difficulty_tasks=difficulty_tasks
        )

    except Exception as e:

        return render_template(
            "index.html",
            error=f"เกิดข้อผิดพลาด: {e}"
        )


# ==========================================
# เริ่มระบบ
# ==========================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )