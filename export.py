"""
Export helpers: turn a list of complaint records into CSV or PDF bytes.
"""
import csv
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

FIELDS = [
    "id",
    "created_at",
    "category",
    "severity",
    "sentiment",
    "summary",
    "root_cause_hypothesis",
    "suggested_action",
    "original_text",
]


def to_csv(records: list[dict]) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=FIELDS, extrasaction="ignore")
    writer.writeheader()
    for r in records:
        writer.writerow(r)
    return buf.getvalue().encode("utf-8")


def to_pdf(records: list[dict]) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    cell_style = ParagraphStyle("cell", parent=styles["Normal"], fontSize=8, leading=10)
    header_style = ParagraphStyle("header", parent=styles["Normal"], fontSize=8, leading=10, textColor=colors.white)

    elements = [Paragraph("Customer Complaint Summary Report", styles["Title"]), Spacer(1, 12)]

    headers = ["ID", "Category", "Severity", "Sentiment", "Summary", "Suggested Action"]
    data = [[Paragraph(h, header_style) for h in headers]]
    for r in records:
        data.append([
            Paragraph(str(r.get("id", "")), cell_style),
            Paragraph(str(r.get("category", "")), cell_style),
            Paragraph(str(r.get("severity", "")), cell_style),
            Paragraph(str(r.get("sentiment", "")), cell_style),
            Paragraph(str(r.get("summary", "")), cell_style),
            Paragraph(str(r.get("suggested_action", "")), cell_style),
        ])

    table = Table(data, colWidths=[30, 90, 55, 65, 260, 220], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2f38")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f4f6")]),
    ]))
    elements.append(table)
    doc.build(elements)
    return buf.getvalue()
