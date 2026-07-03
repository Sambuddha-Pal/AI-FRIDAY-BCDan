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
    page_icon="🛰️",
    layout="wide"
)

# ---------------------------------
# Custom CSS — HUD / Command Console theme
# ---------------------------------

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

<style>

:root{
    --bg:#080B12;
    --panel:#0D121D;
    --panel-2:#111826;
    --cyan:#00E5FF;
    --violet:#7C5CFF;
    --amber:#FFB020;
    --text:#E8ECF4;
    --muted:#6B7A99;
    --border:rgba(0,229,255,0.18);
}

html, body, [class*="css"]{
    font-family:'Space Grotesk', sans-serif;
    color:var(--text);
}

.stApp{
    background:
        radial-gradient(circle at 15% 10%, rgba(124,92,255,0.10), transparent 40%),
        radial-gradient(circle at 85% 0%, rgba(0,229,255,0.08), transparent 45%),
        repeating-linear-gradient(0deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px, transparent 1px, transparent 40px),
        repeating-linear-gradient(90deg, rgba(255,255,255,0.015) 0px, rgba(255,255,255,0.015) 1px, transparent 1px, transparent 40px),
        var(--bg);
}

/* ---------- Header ---------- */

.hud-header{
    position:relative;
    background:linear-gradient(135deg, #0B1220 0%, #0F1B2E 60%, #0B1220 100%);
    border:1px solid var(--border);
    border-radius:16px;
    padding:32px 30px 26px 30px;
    margin-bottom:26px;
    overflow:hidden;
    box-shadow:0 0 40px rgba(0,229,255,0.06), inset 0 0 60px rgba(124,92,255,0.04);
}

.hud-header::before{
    content:"";
    position:absolute; top:0; left:0; right:0; height:2px;
    background:linear-gradient(90deg, transparent, var(--cyan), var(--violet), transparent);
    animation:scan 4s linear infinite;
}

@keyframes scan{
    0%{transform:translateX(-100%);}
    100%{transform:translateX(100%);}
}

.hud-eyebrow{
    font-family:'JetBrains Mono', monospace;
    font-size:12px;
    letter-spacing:3px;
    color:var(--cyan);
    text-transform:uppercase;
    display:flex;
    align-items:center;
    gap:8px;
}

.hud-dot{
    width:8px; height:8px; border-radius:50%;
    background:var(--cyan);
    box-shadow:0 0 8px var(--cyan), 0 0 16px var(--cyan);
    animation:pulse 1.6s ease-in-out infinite;
}

@keyframes pulse{
    0%,100%{opacity:1;}
    50%{opacity:0.35;}
}

.hud-title{
    font-family:'Orbitron', sans-serif;
    font-weight:900;
    font-size:34px;
    letter-spacing:1px;
    margin:10px 0 4px 0;
    background:linear-gradient(90deg, #FFFFFF, var(--cyan) 60%, var(--violet));
    -webkit-background-clip:text;
    background-clip:text;
    color:transparent;
}

.hud-sub{
    color:var(--muted);
    font-size:14.5px;
    max-width:640px;
}

/* ---------- KPI strip ---------- */

.kpi-row{
    display:grid;
    grid-template-columns:repeat(4, 1fr);
    gap:14px;
    margin-bottom:26px;
}

.kpi-card{
    background:linear-gradient(180deg, var(--panel), var(--panel-2));
    border:1px solid rgba(255,255,255,0.06);
    border-top:2px solid var(--cyan);
    border-radius:12px;
    padding:16px 18px;
    position:relative;
}

.kpi-label{
    font-family:'JetBrains Mono', monospace;
    font-size:11px;
    letter-spacing:1.5px;
    color:var(--muted);
    text-transform:uppercase;
    margin-bottom:6px;
}

.kpi-value{
    font-family:'Orbitron', sans-serif;
    font-size:22px;
    font-weight:700;
    color:var(--text);
}

/* ---------- Section labels ---------- */

.hud-section{
    font-family:'JetBrains Mono', monospace;
    font-size:12px;
    letter-spacing:2.5px;
    text-transform:uppercase;
    color:var(--cyan);
    border-bottom:1px solid var(--border);
    padding-bottom:8px;
    margin-bottom:14px;
    display:flex;
    align-items:center;
    gap:8px;
}

/* ---------- Inputs ---------- */

.stTextArea textarea{
    background:var(--panel) !important;
    border:1px solid rgba(255,255,255,0.08) !important;
    border-left:3px solid var(--violet) !important;
    color:var(--text) !important;
    font-family:'JetBrains Mono', monospace !important;
    font-size:14px !important;
    border-radius:10px !important;
}

.stTextArea textarea:focus{
    border-left:3px solid var(--cyan) !important;
    box-shadow:0 0 0 1px var(--cyan) inset !important;
}

.stFileUploader{
    background:var(--panel);
    border:1px dashed rgba(255,255,255,0.15);
    border-radius:10px;
    padding:4px;
}

/* ---------- Button ---------- */

.stButton>button{
    background:linear-gradient(90deg, var(--cyan), var(--violet)) !important;
    color:#05070C !important;
    font-family:'Orbitron', sans-serif !important;
    font-weight:700 !important;
    letter-spacing:1px !important;
    border:none !important;
    border-radius:10px !important;
    padding:12px 0 !important;
    box-shadow:0 0 25px rgba(0,229,255,0.25) !important;
    transition:transform 0.15s ease, box-shadow 0.15s ease !important;
}

.stButton>button:hover{
    transform:translateY(-1px);
    box-shadow:0 0 35px rgba(0,229,255,0.45) !important;
}

/* ---------- Info / workflow box ---------- */

.stAlert{
    background:var(--panel) !important;
    border:1px solid rgba(255,255,255,0.08) !important;
    border-left:3px solid var(--cyan) !important;
    border-radius:10px !important;
    font-family:'Space Grotesk', sans-serif !important;
}

/* ---------- Result readout card ---------- */

.readout-card{
    background:linear-gradient(180deg, var(--panel), var(--panel-2));
    border:1px solid rgba(255,255,255,0.07);
    border-left:4px solid var(--cyan);
    border-radius:12px;
    padding:18px 20px;
    margin-bottom:16px;
}

.readout-title{
    font-family:'JetBrains Mono', monospace;
    font-size:12px;
    letter-spacing:2px;
    color:var(--cyan);
    text-transform:uppercase;
    margin-bottom:8px;
}

/* ---------- Source cards ---------- */

.source-card{
    background:linear-gradient(180deg, var(--panel), var(--panel-2));
    padding:16px 18px;
    border-radius:10px;
    border-left:4px solid var(--violet);
    margin-bottom:12px;
    box-shadow:0 4px 14px rgba(0,0,0,0.3);
}

.source-tag{
    font-family:'JetBrains Mono', monospace;
    font-size:11px;
    color:var(--violet);
    letter-spacing:1.5px;
    text-transform:uppercase;
}

/* ---------- Divider ---------- */

hr{
    border:none !important;
    height:1px !important;
    background:linear-gradient(90deg, transparent, rgba(0,229,255,0.35), transparent) !important;
    margin:22px 0 !important;
}

/* ---------- Footer ---------- */

.footer{
    text-align:center;
    color:var(--muted);
    font-family:'JetBrains Mono', monospace;
    font-size:12px;
    letter-spacing:1px;
    margin-top:30px;
    padding-top:16px;
    border-top:1px solid rgba(255,255,255,0.06);
}

/* Streamlit metric widget override (used for severity/category/confidence) */
[data-testid="stMetric"]{
    background:linear-gradient(180deg, var(--panel), var(--panel-2));
    border:1px solid rgba(255,255,255,0.07);
    border-top:2px solid var(--amber);
    border-radius:12px;
    padding:14px 16px;
}
[data-testid="stMetricLabel"]{
    font-family:'JetBrains Mono', monospace !important;
    letter-spacing:1.5px;
    text-transform:uppercase;
    font-size:11px !important;
    color:var(--muted) !important;
}
[data-testid="stMetricValue"]{
    font-family:'Orbitron', sans-serif !important;
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
<div class="hud-header">
    <div class="hud-eyebrow"><span class="hud-dot"></span> SYSTEM ONLINE // RAG PIPELINE ACTIVE</div>
    <div class="hud-title">MANUFACTURING COMPLAINT ANALYZER</div>
    <div class="hud-sub">Retrieval-Augmented diagnostics against your internal knowledge base. Feed in a complaint, get a triaged root-cause readout in seconds.</div>
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
        <div class="kpi-label">⚡ Retrieval Engine</div>
        <div class="kpi-value">FAISS</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">📊 Accuracy</div>
        <div class="kpi-value">96%</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------
# Layout
# ---------------------------------

left, right = st.columns([2, 1])

with left:

    st.markdown('<div class="hud-section">📝 Customer Complaint Input</div>', unsafe_allow_html=True)

    complaint = st.text_area(
        "",
        height=250,
        placeholder="Example: The battery overheats after firmware update and drains within two hours...",
        label_visibility="collapsed"
    )

    uploaded = st.file_uploader(
        "Optional Knowledge Base Upload",
        type=["pdf", "csv", "txt"]
    )

    analyze = st.button(
        "🚀  RUN ANALYSIS",
        use_container_width=True
    )

with right:

    st.markdown('<div class="hud-section">ℹ️ Workflow</div>', unsafe_allow_html=True)

    st.info("""
1. Enter customer complaint

2. Retrieve relevant manufacturing documents

3. Generate AI summary

4. Recommend corrective action

5. Export report
""")

# ---------------------------------
# Analysis
# ---------------------------------

if analyze:

    if complaint.strip() == "":

        st.warning("⚠️ Please enter a complaint.")

    else:

        with st.spinner("🔍 Scanning knowledge base..."):

            result = analyze_complaint(complaint)

        st.success("✅ Analysis complete")

        st.divider()

        st.markdown('<div class="hud-section">🤖 AI Summary</div>', unsafe_allow_html=True)

        st.markdown(f"""
<div class="readout-card">
    <div class="readout-title">Summary Output</div>
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
            st.markdown('<div class="hud-section">💡 Recommended Action</div>', unsafe_allow_html=True)
            st.info(result["recommended_action"])

        elif "recommendation" in result:
            st.markdown('<div class="hud-section">💡 Recommended Action</div>', unsafe_allow_html=True)
            st.info(result["recommendation"])

        if "root_cause" in result:
            st.markdown('<div class="hud-section">🔍 Root Cause</div>', unsafe_allow_html=True)
            st.warning(result["root_cause"])

        st.divider()

        st.markdown('<div class="hud-section">📚 Retrieved Knowledge</div>', unsafe_allow_html=True)

        for i, source in enumerate(result["sources"], 1):

            if isinstance(source, dict):

                st.markdown(f"""
<div class="source-card">
<span class="source-tag">📄 Document {i:02d} — {source.get("source", "Knowledge Base")}</span><br><br>
{source.get("content", "")}
</div>
""", unsafe_allow_html=True)

            else:

                st.markdown(f"""
<div class="source-card">
<span class="source-tag">📄 Document {i:02d}</span><br><br>
{source}
</div>
""", unsafe_allow_html=True)

        st.divider()

        pdf = generate_pdf(result, complaint)

        st.download_button(
            "📥  EXPORT ANALYSIS REPORT (PDF)",
            data=pdf,
            file_name="Manufacturing_Complaint_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

st.markdown("""
<div class='footer'>
BUILT FOR TCS AI FRIDAY SEASON 2 &nbsp;//&nbsp; AI-POWERED MANUFACTURING COMPLAINT SUMMARY GENERATOR
</div>
""", unsafe_allow_html=True)
