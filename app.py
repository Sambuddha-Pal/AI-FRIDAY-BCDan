from dotenv import load_dotenv
load_dotenv()



import streamlit as st
from core.llm_client import LLMClient
from core.rag_engine import RAGEngine
from core.predictor import check_patient_eligibility, generate_maintenance_schedule
from core.explainer import explain_result

st.set_page_config(page_title="AI-Friday", layout="wide")
st.title("AI-Friday AI System")
st.title("AI-Friday Enterprise AI System")

mode = st.sidebar.selectbox("Select Mode", ["Chat", "Predict", "Explain", "RAG"])
mode = st.sidebar.selectbox(
    "Select Use Case",
    ["Clinical Trial Eligibility", "Maintenance Scheduler"]
)

llm = LLMClient()
rag = RAGEngine()

if mode == "Chat":
    user_input = st.text_input("Ask something:")
    if st.button("Send"):
        response = llm.ask(user_input)
        st.write(response)

elif mode == "RAG":
    query = st.text_input("Ask from knowledge base:")
    if st.button("Search"):
        answer = rag.ask(query, llm)
        st.write(answer)
if mode == "Clinical Trial Eligibility":
    st.subheader("Patient Details")
    age = st.number_input("Age", 0, 100, 45)
    condition = st.text_input("Condition", "diabetes")
    medication = st.text_input("Medication", "metformin")

    if st.button("Check Eligibility"):
        result = check_patient_eligibility(age, condition, medication, rag)
        explanation = explain_result(result, llm)
        st.write("### Result")
        st.json(result)
        st.write("### Explanation")
        st.write(explanation)

elif mode == "Maintenance Scheduler":
    st.subheader("Equipment Details")
    hours_used = st.number_input("Usage Hours Since Last Service", 0, 10000, 1200)
    last_service_days = st.number_input("Days Since Last Service", 0, 1000, 180)

    if st.button("Generate Schedule"):
        result = generate_maintenance_schedule(hours_used, last_service_days, rag)
        explanation = explain_result(result, llm)
        st.write("### Schedule")
        st.json(result)
        st.write("### Explanation")