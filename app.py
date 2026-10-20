import datetime as dt
import json
import uuid
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from predict import predict_oa_risk
try:
    from utils.translations import TRANSLATIONS
    from utils.validation import validate_patient
except ModuleNotFoundError:
    # Fallback for deployments where only app.py and predict.py were uploaded.
    TRANSLATIONS = {
        "English": {},
        "हिन्दी": {"title": "जोड़ स्वास्थ्य स्क्रीनिंग", "patient": "रोगी की जानकारी", "gait": "चाल इनपुट", "results": "परिणाम", "next": "आगे बढ़ें", "save": "रिकॉर्ड सहेजें"},
        "অসমীয়া": {"title": "গাঁঠি স্বাস্থ্য স্ক্ৰীনিং", "patient": "ৰোগীৰ তথ্য", "gait": "খোজৰ ইনপুট", "results": "ফলাফল", "next": "আগবাঢ়ক", "save": "ৰেকৰ্ড সংৰক্ষণ কৰক"},
    }

    def validate_patient(patient: dict) -> list[str]:
        errors = []
        if not patient.get("consent"):
            errors.append("Consent is required before screening.")
        if patient.get("pain_score", 0) > 0 and not patient.get("joints"):
            errors.append("Select at least one painful joint when pain is above zero.")
        if not 18 <= patient.get("age", 0) <= 100:
            errors.append("Age must be between 18 and 100 years.")
        return errors

st.set_page_config(page_title="Joint Health Screening", page_icon="🦵", layout="centered")
ROOT = Path(__file__).parent
RECORDS = ROOT / "records" / "records.jsonl"
GAIT_FEATURES = {
    "gait_speed": ("Gait speed", "m/s", 1.25, 0.20),
    "stride_len": ("Stride length", "m", 1.40, 0.15),
    "cadence": ("Cadence", "steps/min", 110, 10),
    "asymmetry": ("Left-right asymmetry", "%", 4, 3),
    "knee_rom": ("Knee flexion range", "deg", 62, 6),
    "acoustic_rate": ("Joint sound events", "per min", 3, 2.5),
}
SAMPLES = {
    "Example A — typical gait": dict(gait_speed=1.28, stride_len=1.42, cadence=112, asymmetry=3, knee_rom=63, acoustic_rate=2),
    "Example B — mildly altered gait": dict(gait_speed=1.05, stride_len=1.25, cadence=104, asymmetry=8, knee_rom=54, acoustic_rate=7),
    "Example C — markedly altered gait": dict(gait_speed=.80, stride_len=1.05, cadence=95, asymmetry=14, knee_rom=44, acoustic_rate=13),
}
T = TRANSLATIONS
lang = st.sidebar.selectbox("Language / भाषा / ভাষা", list(T))
def tr(key, fallback): return T[lang].get(key, fallback)

ss = st.session_state
ss.setdefault("step", 0); ss.setdefault("patient", None); ss.setdefault("gait", None); ss.setdefault("gait_src", "")

st.title("🦵 " + tr("title", "Joint Health Screening"))
st.caption("Early screening support for osteoarthritis risk — not a diagnosis")
st.warning("This tool is a screening prototype. It cannot diagnose osteoarthritis. Concerning results should be reviewed by a qualified healthcare professional.")
labels = [tr("patient", "1. Patient details"), tr("gait", "2. Gait input"), tr("results", "3. Results")]
ss["step"] = st.radio("Workflow", range(3), format_func=lambda x: labels[x], horizontal=True, index=ss["step"], label_visibility="collapsed")

if ss["step"] == 0:
    with st.form("patient_form"):
        st.subheader("Demographic information")
        c1, c2 = st.columns(2)
        age = c1.number_input("Age (years)", 18, 100, 55)
        sex = c2.selectbox("Sex", ["Female", "Male", "Other", "Prefer not to say"])
        height = c1.number_input("Height (cm)", 100, 220, 158)
        weight = c2.number_input("Weight (kg)", 25, 200, 62)
        st.subheader("Lifestyle and work factors")
        activity = st.select_slider("Physical activity", ["Low", "Moderate", "High"], value="Moderate")
        c1, c2 = st.columns(2)
        squat = c1.selectbox("Frequent squatting / kneeling?", ["No", "Yes"])
        heavy = c2.selectbox("Carrying heavy loads?", ["No", "Yes"])
        past = c1.selectbox("Previous knee or joint injury?", ["No", "Yes"])
        family = c2.selectbox("Family history of joint problems?", ["No", "Yes"])
        st.subheader("Symptoms and mobility")
        joints = st.multiselect("Joints with pain", ["Left knee", "Right knee", "Hip", "Hand/fingers", "Spine", "Ankle"])
        pain = st.slider("Pain level (0 = none, 10 = worst)", 0, 10, 3)
        stiffness = st.number_input("Morning stiffness (minutes)", 0, 240, 10, step=5)
        difficulties = st.multiselect("Difficulty with", ["Climbing stairs", "Standing up from sitting", "Walking 500 m", "Squatting"])
        c1, c2, c3 = st.columns(3)
        swelling, crepitus, night = c1.checkbox("Joint swelling"), c2.checkbox("Cracking/grinding"), c3.checkbox("Night pain")
        consent = st.checkbox("Patient agreed to anonymous screening and data recording")
        submitted = st.form_submit_button(tr("next", "Save and continue"), type="primary")
    if submitted:
        patient = dict(age=age, sex=sex, height=height, weight=weight, bmi=round(weight/(height/100)**2, 1), activity=activity, squatting=squat, heavy_load=heavy, past_injury=past, family_history=family, joints=joints, pain_score=pain, stiffness_min=stiffness, difficulties=difficulties, swelling=swelling, crepitus=crepitus, night_pain=night, consent=consent)
        errors = validate_patient(patient)
        if errors:
            for error in errors: st.error(error)
        else:
            ss["patient"] = patient; ss["step"] = 1; st.rerun()

elif ss["step"] == 1:
    if not ss["patient"]: st.warning("Complete the patient form first.")
    else:
        st.subheader("Gait and movement input")
        st.caption("Sensor sleeve data can be connected here later through the Raspberry Pi interface.")
        source = st.radio("Data source", ["Skip gait input", "Sensor sleeve — Raspberry Pi", "Development sample participant", "Upload feature vector JSON"])
        gait = None
        if source == "Sensor sleeve — Raspberry Pi": st.info("Hardware interface placeholder. Connect the Raspberry Pi feature endpoint here later.")
        elif source == "Development sample participant":
            name = st.selectbox("Sample participant", list(SAMPLES)); gait = SAMPLES[name]
            st.warning("Dataset demonstration only — this is not an assessment of the patient.")
        elif source == "Upload feature vector JSON":
            uploaded = st.file_uploader("Upload JSON feature vector", type="json")
            if uploaded:
                try:
                    gait = json.load(uploaded); missing = [k for k in GAIT_FEATURES if k not in gait]
                    if missing: st.error("Missing keys: " + ", ".join(missing)); gait = None
                except json.JSONDecodeError: st.error("The file is not valid JSON.")
            st.warning("Development feature vector only — not a patient assessment.")
        if gait:
            st.dataframe(pd.DataFrame([(GAIT_FEATURES[k][0], gait[k], GAIT_FEATURES[k][1]) for k in GAIT_FEATURES], columns=["Measure", "Value", "Unit"]), hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        if c1.button("← Back"): ss["step"] = 0; st.rerun()
        if c2.button(tr("next", "Continue to results"), type="primary"): ss["gait"] = gait; ss["gait_src"] = source; ss["step"] = 2; st.rerun()

else:
    if not ss["patient"]: st.warning("Complete the patient form first.")
    else:
        patient, gait = ss["patient"], ss["gait"]; result = predict_oa_risk(patient, gait)
        colors = {"Low":"#2e9e5b", "Moderate":"#d09217", "High":"#c94b4b"}; colour = colors[result["risk_level"]]
        if result.get("is_placeholder"): st.warning("Development mode: this result uses a placeholder prediction function.")
        c1, c2 = st.columns(2)
        c1.metric("Screening risk level", result["risk_level"]); c1.write("Indicative severity: **" + result["severity"] + "**")
        fig = go.Figure(go.Indicator(mode="gauge+number", value=round(result["risk_score"]*100), number={"suffix":"%"}, gauge={"axis":{"range":[0,100]},"bar":{"color":colour},"steps":[{"range":[0,33],"color":"#e6f4ec"},{"range":[33,60],"color":"#fbf1d3"},{"range":[60,100],"color":"#f8dcdc"}]})); fig.update_layout(height=210, margin=dict(l=10,r=10,t=10,b=0)); c2.plotly_chart(fig, use_container_width=True)
        st.info({"Low":"Share joint-care advice and consider re-screening in 6–12 months.","Moderate":"Refer to the nearest doctor or PHC for a clinical check.","High":"Refer to a doctor for clinical assessment soon."}[result["risk_level"]])
        tab1, tab2, tab3 = st.tabs(["Why this result", "Gait comparison", "Input summary"])
        with tab1:
            factors = pd.Series(result["factors"]).sort_values(); bar = go.Figure(go.Bar(x=factors.values, y=factors.index, orientation="h", marker_color=colour)); bar.update_layout(height=320, xaxis_title="Contribution to screening score", margin=dict(l=0,r=0,t=10,b=0)); st.plotly_chart(bar, use_container_width=True)
        with tab2:
            if gait:
                z = [(gait[k]-meta[2])/meta[3] for k,meta in GAIT_FEATURES.items()]; chart = go.Figure(go.Bar(x=[meta[0] for meta in GAIT_FEATURES.values()], y=z)); chart.update_layout(height=320, yaxis_title="Difference from reference (SD)"); st.plotly_chart(chart, use_container_width=True)
            else: st.write("No gait data was added. The result uses questionnaire inputs only.")
        with tab3: st.dataframe(pd.DataFrame([(k, ", ".join(v) if isinstance(v,list) else v) for k,v in patient.items() if k != "consent"], columns=["Input","Value"]), hide_index=True, use_container_width=True)
        st.error("This is a screening result, NOT a medical diagnosis.")
        c1, c2 = st.columns(2)
        if c1.button("← Review inputs"): ss["step"] = 1; st.rerun()
        if c2.button(tr("save", "Save screening record"), type="primary"):
            RECORDS.parent.mkdir(exist_ok=True); record = {"id":uuid.uuid4().hex[:8],"time":dt.datetime.now().isoformat(timespec="seconds"),"patient":patient,"gait":gait,"gait_is_demo":"Development" in ss["gait_src"],"result":{k:v for k,v in result.items() if k != "factors"},"synced":False}; RECORDS.open("a").write(json.dumps(record)+"\n"); st.success("Saved on this device. Code: " + record["id"])

st.sidebar.divider(); count = sum(1 for _ in RECORDS.open()) if RECORDS.exists() else 0; st.sidebar.metric("Records on this device", count); st.sidebar.caption("Offline records remain on this device until exported or synced.")
if RECORDS.exists(): st.sidebar.download_button("Export records", RECORDS.read_bytes(), "oa_records.jsonl")

