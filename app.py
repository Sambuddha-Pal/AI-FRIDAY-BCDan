import streamlit as st
import pandas as pd
import plotly.express as px

# ----------------------------------------
# Page Configuration
# ----------------------------------------
st.set_page_config(
    page_title="AI Customer Complaint Summary Generator",
    page_icon="🏭",
    layout="wide"
)

# ----------------------------------------
# Sidebar
# ----------------------------------------
st.sidebar.title("🏭 Manufacturing AI")

page = st.sidebar.radio(
    "Navigation",
    [
        "Complaint Analysis",
        "Bulk Upload",
        "Analytics",
        "History",
        "Settings"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **TCS AI Friday 2026**

    AI-Powered Customer Complaint
    Summary Generator
    """
)

# ----------------------------------------
# Dummy AI Function
# Replace with OpenAI API
# ----------------------------------------
def analyze_complaint(text):
    return {
        "Summary": "Customer reports overheating and rapid battery drain after firmware update.",
        "Category": "Battery",
        "Severity": "High",
        "Sentiment": "Negative",
        "Confidence": "96%",
        "Recommended Action": "Escalate to firmware engineering team."
    }

# ========================================
# Complaint Analysis
# ========================================

if page == "Complaint Analysis":

    st.title("🏭 AI Complaint Summary Generator")

    col1, col2 = st.columns([2, 1])

    with col1:

        complaint = st.text_area(
            "Customer Complaint",
            height=250,
            placeholder="Paste customer complaint here..."
        )

    with col2:

        st.metric("Today's Complaints", "128")
        st.metric("AI Accuracy", "96%")
        st.metric("Time Saved", "78%")

    if st.button("🚀 Generate AI Summary", use_container_width=True):

        if complaint.strip():

            with st.spinner("Analyzing Complaint..."):

                result = analyze_complaint(complaint)

            st.success("Analysis Complete")

            st.subheader("📋 AI Summary")

            st.write(result["Summary"])

            c1, c2, c3 = st.columns(3)

            c1.metric("Category", result["Category"])
            c2.metric("Severity", result["Severity"])
            c3.metric("Sentiment", result["Sentiment"])

            st.metric("Confidence", result["Confidence"])

            st.subheader("Recommended Action")

            st.info(result["Recommended Action"])

            st.subheader("Edit Summary")

            edited = st.text_area(
                "",
                value=result["Summary"],
                height=120
            )

            st.button("Save Edited Summary")

# ========================================
# Bulk Upload
# ========================================

elif page == "Bulk Upload":

    st.title("📂 Bulk Complaint Upload")

    uploaded = st.file_uploader(
        "Upload CSV or JSON",
        type=["csv", "json"]
    )

    if uploaded:

        if uploaded.name.endswith(".csv"):

            df = pd.read_csv(uploaded)

        else:

            df = pd.read_json(uploaded)

        st.success("File Uploaded")

        st.dataframe(df)

        if st.button("Analyze All Complaints"):

            st.success("Processing Complete")

# ========================================
# Analytics
# ========================================

elif page == "Analytics":

    st.title("📊 Complaint Analytics")

    df = pd.DataFrame({
        "Category": [
            "Battery",
            "Display",
            "Audio",
            "Battery",
            "Camera"
        ],
        "Count": [40, 20, 12, 15, 18]
    })

    fig = px.bar(
        df,
        x="Category",
        y="Count",
        color="Category",
        title="Complaint Categories"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("KPIs")

    c1, c2, c3 = st.columns(3)

    c1.metric("Complaints", "1520")
    c2.metric("Resolved", "1425")
    c3.metric("Critical", "32")

# ========================================
# History
# ========================================

elif page == "History":

    st.title("📜 Complaint History")

    history = pd.DataFrame({

        "Complaint": [
            "Battery drains quickly",
            "Screen flickers",
            "Speaker not working"
        ],

        "Category": [
            "Battery",
            "Display",
            "Audio"
        ],

        "Severity": [
            "High",
            "Medium",
            "Low"
        ]

    })

    st.dataframe(history, use_container_width=True)

# ========================================
# Settings
# ========================================

elif page == "Settings":

    st.title("⚙ Settings")

    st.text_input("OpenAI API Key", type="password")

    st.selectbox(
        "Model",
        [
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-5"
        ]
    )

    st.slider(
        "Temperature",
        0.0,
        1.0,
        0.3
    )

    st.button("Save Settings")