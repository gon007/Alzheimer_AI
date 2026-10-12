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
    education_level,
    bmi,
    diabetes,
    hypertension,
    family_history,
    cognitive_score
):

    features = np.array(
        [[
            age,
            gender,
            education_level,
            bmi,
            diabetes,
            hypertension,
            family_history,
            cognitive_score
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

        education_level = float(
            request.form.get(
                "education_level",
                0
            )
        )

        bmi = float(
            request.form.get(
                "bmi",
                0
            )
        )

        diabetes = int(
            request.form.get(
                "diabetes",
                0
            )
        )

        hypertension = int(
            request.form.get(
                "hypertension",
                0
            )
        )

        family_history = int(
            request.form.get(
                "family_history",
                0
            )
        )

        cognitive_score = float(
            request.form.get(
                "cognitive_score",
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
        # แปลงเพศ
        # -----------------------------

        if gender_text == "male":

            gender = 0

        else:

            gender = 1


        # -----------------------------
        # ตรวจสอบระดับการศึกษา
        # -----------------------------

        if education_level < 0 or education_level > 3:

            return render_template(
                "index.html",
                error="ระดับการศึกษาต้องอยู่ระหว่าง 0-3"
            )


        # -----------------------------
        # ตรวจสอบ BMI
        # -----------------------------

        if bmi < 10 or bmi > 60:

            return render_template(
                "index.html",
                error="กรุณากรอกค่า BMI ระหว่าง 10-60"
            )


        # -----------------------------
        # ตรวจสอบข้อมูล Yes / No
        # -----------------------------

        answers = [
            diabetes,
            hypertension,
            family_history
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
        # ตรวจสอบ Cognitive Score
        # -----------------------------

        if cognitive_score < 0 or cognitive_score > 100:

            return render_template(
                "index.html",
                error="คะแนน Cognitive Test ต้องอยู่ระหว่าง 0-100"
            )


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
            education_level,
            bmi,
            diabetes,
            hypertension,
            family_history,
            cognitive_score
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
                "ควรติดตามสุขภาพและการเปลี่ยนแปลงด้านการรู้คิด "
                "และปรึกษาผู้เชี่ยวชาญหากมีความกังวล"
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

            education_level=int(education_level),

            bmi=bmi,

            diabetes=diabetes,

            hypertension=hypertension,

            family_history=family_history,

            cognitive_score=int(cognitive_score)
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