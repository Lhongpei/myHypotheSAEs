import streamlit as st
import pandas as pd
import json

# 读取数据
def load_patients(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

patients = load_patients("patients.jsonl")

# 选择患者
patient_ids = [p["id"] for p in patients]
selected_id = st.selectbox("🔍 选择患者 ID", patient_ids)

patient = next(p for p in patients if p["id"] == selected_id)

# 展示 profile
st.markdown("## 🧬 Profile")
for k, v in patient["profile"].items():
    st.markdown(f"**{k}**: {v}")

# 展示原始数据表格
st.markdown("## 📑 原始数据")
df = pd.DataFrame(patient["table"])
st.dataframe(df)

# 展示 hypotheses
st.markdown("## 🧠 Hypotheses")
for i, h in enumerate(patient["hypotheses"]):
    with st.expander(f"Hypothesis {i+1}"):
        st.markdown(h)
