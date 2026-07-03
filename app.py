import streamlit as st
from backend.rag import analyze_complaint

from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

# ---------------------------------
# Page Config
# ---------------------------------

st.set_page_config(
    page_title="AI Manufacturing Complaint Analyzer",
    page_icon="⚙️",
    layout="wide"
)

# ---------------------------------
# Custom CSS — Glass-panel industrial console
# ---------------------------------

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700;800&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>

:root{
    --bg-1:#14181F;
    --bg-2:#1B2029;
    --glass:rgba(255,255,255,0.045);
    --glass-border:rgba(255,255,255,0.09);
    --teal:#3DDC97;
    --amber:#FFA23A;
    --red:#FF5C5C;
    --text:#EDEFF3;
    --muted:#8B93A3;
}

html, body, [class*="css"]{
    font-family:'Inter', sans-serif;
    color:var(--text);
}

.stApp{
    background:
        radial-gradient(ellipse at top left, rgba(61,220,151,0.06), transparent 50%),
        radial-gradient(ellipse at bottom right, rgba(255,162,58,0.05), transparent 50%),
        linear-gradient(180deg, var(--bg-1), var(--bg-2));
}

/* ---------- Header ---------- */

.panel-header{
    background:var(--glass);
    backdrop-filter:blur(14px);
    border:1px solid var(--glass-border);
    border-radius:20px;
    padding:30px 32px;
    margin-bottom:24px;
    display:flex;
    align-items:center;
    justify-content:space-between;
    flex-wrap:wrap;
    gap:16px;
}

.panel-title{
    font-family:'Sora', sans-serif;
    font-weight:800;
    font-size:28px;
    color:var(--text);
    margin:0;
}

.panel-sub{
    color:var(--muted);
    font-size:14px;
    margin-top:4px;
    max-width:520px;
}

.status-pill{
    background:rgba(61,220,151,0.12);
    border:1px solid rgba(61,220,151,0.35);
    color:var(--teal);
    font-family:'IBM Plex Mono', monospace;
    font-size:12px;
    letter-spacing:0.5px;
    padding:8px 14px;
    border-radius:100px;
    display:flex;
    align-items:center;
    gap:8px;
    white-space:nowrap;
}

.status-pill .dot{
    width:7px; height:7px; border-radius:50%;
    background:var(--teal);
    box-shadow:0 0 6px var(--teal);
}

/* ---------- KPI strip ---------- */

.kpi-row{
    display:grid;
    grid-template-columns:repeat(4, 1fr);
    gap:14px;
    margin-bottom:24px;
}

.kpi-card{
    background:var(--glass);
    backdrop-filter:blur(10px);
    border:1px solid var(--glass-border);
    border-radius:16px;
    padding:18px 20px;
    transition:border-color 0.2s ease;
}

.kpi-card:hover{
    border-color:rgba(61,220,151,0.4);
}

.kpi-label{
    font-size:12.5px;
    color:var(--muted);
    margin-bottom:8px;
    font-weight:500;
}

.kpi-value{
    font-family:'Sora', sans-serif;
    font-size:21px;
    font-weight:700;
    color:var(--text);
}

/* ---------- Panel / section wrapper ---------- */

.panel{
    background:var(--glass);
    backdrop-filter:blur(10px);
    border:1px solid var(--glass-border);
    border-radius:18px;
    padding:24px 26px;
    margin-bottom:20px;
}

.panel-heading{
    font-family:'Sora', sans-serif;
    font-weight:700;
    font-size:15px;
    color:var(--text);
    margin-bottom:14px;
    display:flex;
    align-items:center;
    gap:8px;
}

/* ---------- Inputs ---------- */

.stTextArea textarea{
    background:rgba(0,0,0,0.18) !important;
    border:1px solid var(--glass-border) !important;
    color:var(--text) !important;
    border-radius:12px !important;
    font-size:14.5px !important;
}

.stTextArea textarea:focus{
    border-color:var(--teal) !important;
    box-shadow:0 0 0 1px var(--teal) inset !important;
}

.stFileUploader{
    background:rgba(0,0,0,0.15);
    border:1px dashed var(--glass-border);
    border-radius:12px;
}

/* ---------- Button ---------- */

.stButton>button{
    background:linear-gradient(135deg, var(--teal), #2FBF83) !important;
    color:#0C1410 !important;
    font-family:'Sora', sans-serif !important;
    font-weight:700 !important;
    border:none !important;
    border-radius:12px !important;
    padding:13px 0 !important;
    box-shadow:0 6px 20px rgba(61,220,151,0.22) !important;
    transition:transform 0.15s ease !important;
}

.stButton>button:hover{
    transform:translateY(-1px);
}

/* ---------- Alerts / info boxes ---------- */

.stAlert{
    background:rgba(0,0,0,0.18) !important;
    border:1px solid var(--glass-border) !important;
    border-left:3px solid var(--amber) !important;
    border-radius:12px !important;
}

/* ---------- Result readout ---------- */

.readout{
    background:rgba(61,220,151,0.06);
    border:1px solid rgba(61,220,151,0.25);
    border-radius:14px;
    padding:18px 20px;
    margin-bottom:16px;
    line-height:1.55;
}

.readout-label{
    font-family:'IBM Plex Mono', monospace;
    font-size:11.5px;
    letter-spacing:1px;
    text-transform:uppercase;
    color:var(--teal);
    margin-bottom:8px;
}

/* ---------- Source cards ---------- */

.source-card{
    background:rgba(255,255,255,0.03);
    border:1px solid var(--glass-border);
    border-radius:12px;
    padding:16px 18px;
    margin-bottom:12px;
}

.source-tag{
    font-family:'IBM Plex Mono', monospace;
    font-size:11.5px;
    color:var(--amber);
    letter-spacing:0.5px;
}

/* ---------- Divider ---------- */

hr{
    border:none !important;
    height:1px !important;
    background:var(--glass-border) !important;
    margin:22px 0 !important;
}

/* ---------- Footer ---------- */

.footer{
    text-align:center;
    color:var(--muted);
    font-size:12.5px;
    margin-top:28px;
    padding-top:16px;
    border-top:1px solid var(--glass-border);
}

/* Streamlit metric widget override */
[data-testid="stMetric"]{
    background:var(--glass);
    backdrop-filter:blur(10px);
    border:1px solid var(--glass-border);
    border-radius:14px;
    padding:14px 16px;
}
[data-testid="stMetricLabel"]{
    color:var(--muted) !important;
    font-size:12px !important;
}
[data-testid="stMetricValue"]{
    font-family:'Sora', sans-serif !important;
    color:var(--text) !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------
# PDF (unchanged logic)
# ---------------------------------

def generate_pdf(result, complaint):

    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph("<b>Manufacturing Complaint Analysis Report</b>", styles["Title"]))

    story.append(Paragraph("<b>Customer Complaint</b>", styles["Heading2"]))
    story.append(Paragraph(complaint, styles["BodyText"]))

    story.append(Paragraph("<b>Summary</b>", styles["Heading2"]))
    story.append(Paragraph(result["summary"], styles["BodyText"]))

    story.append(Paragraph("<b>Category</b>", styles["Heading2"]))
    story.append(Paragraph(result["category"], styles["BodyText"]))

    story.append(Paragraph("<b>Severity</b>", styles["Heading2"]))
    story.append(Paragraph(result["severity"], styles["BodyText"]))

    if "recommended_action" in result:
        story.append(Paragraph("<b>Recommended Action</b>", styles["Heading2"]))
        story.append(Paragraph(result["recommended_action"], styles["BodyText"]))

    doc.build(story)

    buffer.seek(0)

    return buffer

# ---------------------------------
# Header
# ---------------------------------

st.markdown("""
<div class="panel-header">
    <div>
        <p class="panel-title">⚙️ AI Manufacturing Complaint Analyzer</p>
        <p class="panel-sub">Retrieval-Augmented Generation over your internal knowledge base — summary, severity, and corrective action in one pass.</p>
    </div>
    <div class="status-pill"><span class="dot"></span> RAG ENGINE READY</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------
# KPI Cards
# ---------------------------------

st.markdown("""
<div class="kpi-row">
    <div class="kpi-card">
        <div class="kpi-label">📄 Knowledge Docs</div>
        <div class="kpi-value">250+</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">🤖 AI Model</div>
        <div class="kpi-value">GPT-3.5</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">⚡ Retrieval</div>
        <div class="kpi-value">FAISS</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">📊 Accuracy</div>
        <div class="kpi-value">96%</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------
# Layout
# ---------------------------------

left, right = st.columns([2, 1])

with left:

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">📝 Customer Complaint</div>', unsafe_allow_html=True)

    complaint = st.text_area(
        "",
        height=230,
        placeholder="Example: The battery overheats after firmware update and drains within two hours...",
        label_visibility="collapsed"
    )

    uploaded = st.file_uploader(
        "Optional Knowledge Base Upload",
        type=["pdf", "csv", "txt"]
    )

    analyze = st.button(
        "🚀 Analyze Complaint",
        use_container_width=True
    )

    st.markdown('</div>', unsafe_allow_html=True)

with right:

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel-heading">ℹ️ Workflow</div>', unsafe_allow_html=True)

    st.info("""
1. Enter customer complaint

2. Retrieve relevant manufacturing documents

3. Generate AI summary

4. Recommend corrective action

5. Export report
""")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------
# Analysis
# ---------------------------------

if analyze:

    if complaint.strip() == "":

        st.warning("Please enter a complaint.")

    else:

        with st.spinner("🔍 Searching Knowledge Base..."):

            result = analyze_complaint(complaint)

        st.success("Analysis Completed")

        st.divider()

        st.markdown('<div class="panel-heading">🤖 AI Summary</div>', unsafe_allow_html=True)

        st.markdown(f"""
<div class="readout">
    <div class="readout-label">Summary</div>
    {result["summary"]}
</div>
""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)

        c1.metric("Category", result["category"])
        c2.metric("Severity", result["severity"])

        if "confidence" in result:
            c3.metric("Confidence", result["confidence"])
        else:
            c3.metric("Status", "Completed")

        if "recommended_action" in result:
            st.markdown('<div class="panel-heading">💡 Recommended Action</div>', unsafe_allow_html=True)
            st.info(result["recommended_action"])

        elif "recommendation" in result:
            st.markdown('<div class="panel-heading">💡 Recommended Action</div>', unsafe_allow_html=True)
            st.info(result["recommendation"])

        if "root_cause" in result:
            st.markdown('<div class="panel-heading">🔍 Root Cause</div>', unsafe_allow_html=True)
            st.warning(result["root_cause"])

        st.divider()

        st.markdown('<div class="panel-heading">📚 Retrieved Knowledge</div>', unsafe_allow_html=True)

        for i, source in enumerate(result["sources"], 1):

            if isinstance(source, dict):

                st.markdown(f"""
<div class="source-card">
<span class="source-tag">📄 Document {i} · {source.get("source", "Knowledge Base")}</span><br><br>
{source.get("content", "")}
</div>
""", unsafe_allow_html=True)

            else:

                st.markdown(f"""
<div class="source-card">
<span class="source-tag">📄 Document {i}</span><br><br>
{source}
</div>
""", unsafe_allow_html=True)

        st.divider()

        pdf = generate_pdf(result, complaint)

        st.download_button(
            "📥 Export Analysis Report (PDF)",
            data=pdf,
            file_name="Manufacturing_Complaint_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

st.markdown("""
<div class='footer'>
Built for TCS AI Friday Season 2 · AI-Powered Manufacturing Complaint Summary Generator
</div>
""", unsafe_allow_html=True)
