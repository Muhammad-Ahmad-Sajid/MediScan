
report_code = """
def generate_fracture_report(scan, patient, prognosis=None, output_dir="reports/"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        patient_name_safe = "".join([c if c.isalnum() else "_" for c in patient.get("full_name", "Unknown")])
        scan_id = getattr(scan, "id", scan.get("id", "unknown")) if not isinstance(scan, dict) else scan.get("id", "unknown")
        filename = f"fracture_{scan_id}_{patient_name_safe}_{date_str}.pd"
        filepath = os.path.join(output_dir, filename).replace("\\\\", "/")
        
        report_id = str(uuid.uuid4())[:8].upper()
        report_date = datetime.now().strftime("%Y-%m-%d")
        report_time = datetime.now().strftime("%H:%M:%S")

        doc = create_report(filepath, report_id, report_date, report_time)
        elements = []

        styles = getSampleStyleSheet()
        style_h2 = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=12, spaceAfter=8, textTransform="uppercase", textColor=colors.HexColor("#333333"))
        style_normal = styles["Normal"]
        style_normal.fontName = "Helvetica"

        # TITLE
        elements.append(Paragraph("<b>MediScan AI &mdash; Fracture Detection Report</b>", styles["Title"]))
        elements.append(Spacer(1, 0.2 * inch))

        # PATIENT INFO
        elements.append(Paragraph("PATIENT INFORMATION", style_h2))
        patient_data = [
            [Paragraph(f"<b>Name:</b> {patient.get('full_name', 'N/A')}", style_normal), Paragraph(f"<b>Patient ID:</b> {patient.get('id', 'N/A')}", style_normal)],
            [Paragraph(f"<b>Age:</b> {patient.get('age', 'N/A')}", style_normal), Paragraph(f"<b>Gender:</b> {patient.get('gender', 'N/A')}", style_normal)],
        ]
        t_patient = Table(patient_data, colWidths=[3.6 * inch, 3.6 * inch])
        t_patient.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        elements.append(t_patient)
        elements.append(Spacer(1, 0.2 * inch))
        
        # SCAN DETAILS
        elements.append(Paragraph("SCAN DETAILS", style_h2))
        scan_dt = getattr(scan, "upload_timestamp", None)
        scan_dt_str = scan_dt.strftime("%Y-%m-%d %H:%M:%S") if scan_dt else "N/A"
        scan_region = getattr(scan, "body_region", "N/A")
        if not isinstance(scan, dict):
            scan_region = getattr(scan, "body_region", "N/A")
        else:
            scan_region = scan.get("body_region", "N/A")
            
        elements.append(Paragraph(f"<b>Scan Date:</b> {scan_dt_str}", style_normal))
        elements.append(Paragraph(f"<b>Body Region:</b> {scan_region}", style_normal))
        elements.append(Spacer(1, 0.2 * inch))

        # DIAGNOSIS
        elements.append(Paragraph("DIAGNOSIS SECTION", style_h2))
        frac_detected = getattr(scan, "fracture_detected", False)
        badge_text = "<font color='white'><b>FRACTURE DETECTED</b></font>" if frac_detected else "<font color='white'><b>NO FRACTURE DETECTED</b></font>"
        badge_bg = "#D32F2F" if frac_detected else "#388E3C"
        
        badge = Table([[Paragraph(badge_text, ParagraphStyle("Centered", parent=style_normal, alignment=TA_CENTER))]],
                      colWidths=[2.5 * inch], rowHeights=[0.3 * inch],
                      style=[("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(badge_bg)), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")])
        elements.append(badge)
        elements.append(Spacer(1, 0.1 * inch))
        
        conf = getattr(scan, "confidence", 0.0)
        conf_flag = getattr(scan, "confidence_flag", "low")
        elements.append(Paragraph(f"<b>Prediction:</b> {'Fractured' if frac_detected else 'Not Fractured'}", style_normal))
        elements.append(Paragraph(f"<b>Confidence:</b> {conf:.1f}% ({conf_flag})", style_normal))
        elements.append(Spacer(1, 0.2 * inch))

        # IMAGES
        img_path = getattr(scan, "original_file_path", "")
        heatmap_path = getattr(scan, "heatmap_path", "")
        img_table = Table([[get_image_flowable(img_path, "Original X-Ray", style_normal), get_image_flowable(heatmap_path, "Grad-CAM Heatmap", style_normal)]])
        elements.append(img_table)
        elements.append(Spacer(1, 0.2 * inch))

        # PROGNOSIS
        if prognosis:
            elements.append(Paragraph("PROGNOSIS & RECOMMENDATIONS", style_h2))
            prog_data = [
                [Paragraph("<b>Estimated Healing Time:</b>", style_normal), Paragraph(f"{prognosis.rest_weeks_min} - {prognosis.rest_weeks_max} weeks", style_normal)],
                [Paragraph("<b>Cast Type:</b>", style_normal), Paragraph(str(prognosis.cast_type), style_normal)],
                [Paragraph("<b>Weight Bearing:</b>", style_normal), Paragraph(str(prognosis.weight_bearing_status), style_normal)],
            ]
            t_prog = Table(prog_data, colWidths=[2 * inch, 5 * inch])
            elements.append(t_prog)
            
            if prognosis.referral_flag:
                elements.append(Spacer(1, 0.1 * inch))
                elements.append(Paragraph("<font color='red'><b>SURGERY REFERRAL RECOMMENDED</b></font>", style_normal))
            elements.append(Spacer(1, 0.2 * inch))

        # OVERRIDE
        override = getattr(scan, "clinician_override", None)
        if override:
            elements.append(Paragraph("CLINICIAN OVERRIDE", style_h2))
            elements.append(Paragraph(f"<b>Override Diagnosis:</b> {override}", style_normal))
            elements.append(Paragraph(f"<b>Notes:</b> {getattr(scan, 'override_notes', 'N/A')}", style_normal))
            elements.append(Spacer(1, 0.2 * inch))

        # RECOMMENDATION
        elements.append(Paragraph("CLINICAL RECOMMENDATION", style_h2))
        if conf_flag == "clear":
            elements.append(Paragraph("Clear finding, clinical correlation recommended.", style_normal))
        elif conf_flag == "inconclusive":
            elements.append(Paragraph("Inconclusive &mdash; manual radiologist review required.", style_normal))
        else:
            elements.append(Paragraph("Low confidence &mdash; recommend repeat imaging or specialist review.", style_normal))

        # BUILD
        build_pdf(doc, elements)
        return filepath
    except Exception as e:
        logger.error(f"Error generating fracture report: {e}")
        return None

def generate_arthritis_report(scan, patient, output_dir="reports/"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        patient_name_safe = "".join([c if c.isalnum() else "_" for c in patient.get("full_name", "Unknown")])
        scan_id = getattr(scan, "id", scan.get("id", "unknown")) if not isinstance(scan, dict) else scan.get("id", "unknown")
        filename = f"arthritis_{scan_id}_{patient_name_safe}_{date_str}.pd"
        filepath = os.path.join(output_dir, filename).replace("\\\\", "/")
        
        report_id = str(uuid.uuid4())[:8].upper()
        report_date = datetime.now().strftime("%Y-%m-%d")
        report_time = datetime.now().strftime("%H:%M:%S")

        doc = create_report(filepath, report_id, report_date, report_time)
        elements = []

        styles = getSampleStyleSheet()
        style_h2 = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=12, spaceAfter=8, textTransform="uppercase", textColor=colors.HexColor("#333333"))
        style_normal = styles["Normal"]
        style_normal.fontName = "Helvetica"

        # TITLE
        elements.append(Paragraph("<b>MediScan AI &mdash; Arthritis Grading Report</b>", styles["Title"]))
        elements.append(Spacer(1, 0.2 * inch))

        # PATIENT INFO
        elements.append(Paragraph("PATIENT INFORMATION", style_h2))
        patient_data = [
            [Paragraph(f"<b>Name:</b> {patient.get('full_name', 'N/A')}", style_normal), Paragraph(f"<b>Patient ID:</b> {patient.get('id', 'N/A')}", style_normal)],
            [Paragraph(f"<b>Age:</b> {patient.get('age', 'N/A')}", style_normal), Paragraph(f"<b>Gender:</b> {patient.get('gender', 'N/A')}", style_normal)],
        ]
        t_patient = Table(patient_data, colWidths=[3.6 * inch, 3.6 * inch])
        t_patient.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        elements.append(t_patient)
        elements.append(Spacer(1, 0.2 * inch))
        
        # DIAGNOSIS
        elements.append(Paragraph("DIAGNOSIS SECTION", style_h2))
        grade = getattr(scan, "grade", 0)
        grade_name = getattr(scan, "grade_name", "Normal")
        
        colors_map = {0: "#388E3C", 1: "#81C784", 2: "#FBC02D", 3: "#F57C00", 4: "#D32F2F"}
        badge_bg = colors_map.get(grade, "#333333")
        badge_text = f"<font color='white'><b>Grade {grade} - {grade_name}</b></font>"
        
        badge = Table([[Paragraph(badge_text, ParagraphStyle("Centered", parent=style_normal, alignment=TA_CENTER))]],
                      colWidths=[2.5 * inch], rowHeights=[0.3 * inch],
                      style=[("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(badge_bg)), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")])
        elements.append(badge)
        elements.append(Spacer(1, 0.1 * inch))
        
        conf = getattr(scan, "confidence", 0.0)
        elements.append(Paragraph(f"<b>Confidence:</b> {conf:.1f}%", style_normal))
        elements.append(Spacer(1, 0.2 * inch))

        # IMAGES
        img_path = getattr(scan, "original_file_path", "")
        heatmap_path = getattr(scan, "heatmap_path", "")
        img_table = Table([[get_image_flowable(img_path, "Original X-Ray", style_normal), get_image_flowable(heatmap_path, "Grad-CAM Heatmap", style_normal)]])
        elements.append(img_table)
        elements.append(Spacer(1, 0.2 * inch))
        
        # GRADE TABLE
        elements.append(Paragraph("GRADE EXPLANATION", style_h2))
        grade_desc = [
            ("Grade 0", "Normal", "No joint space narrowing or reactive changes."),
            ("Grade 1", "Doubtful", "Doubtful joint space narrowing and possible osteophytic lipping."),
            ("Grade 2", "Mild", "Definite osteophytes and possible joint space narrowing."),
            ("Grade 3", "Moderate", "Moderate multiple osteophytes, definite narrowing, sclerosis."),
            ("Grade 4", "Severe", "Large osteophytes, marked narrowing, severe sclerosis, deformity.")
        ]
        t_data = []
        for g_num, g_name, g_txt in grade_desc:
            prefix = "<b>" if f"Grade {grade}" == g_num else ""
            suffix = "</b>" if f"Grade {grade}" == g_num else ""
            t_data.append([Paragraph(f"{prefix}{g_num}{suffix}", style_normal), Paragraph(f"{prefix}{g_name}{suffix}", style_normal), Paragraph(f"{prefix}{g_txt}{suffix}", style_normal)])
            
        t_grades = Table(t_data, colWidths=[1.5*inch, 1.5*inch, 4*inch])
        t_grades.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.5, colors.grey)]))
        elements.append(t_grades)
        elements.append(Spacer(1, 0.2 * inch))

        # OVERRIDE
        override = getattr(scan, "clinician_override", None)
        if override:
            elements.append(Paragraph("CLINICIAN OVERRIDE", style_h2))
            elements.append(Paragraph(f"<b>Override Diagnosis:</b> {override}", style_normal))
            elements.append(Paragraph(f"<b>Notes:</b> {getattr(scan, 'override_notes', 'N/A')}", style_normal))
            elements.append(Spacer(1, 0.2 * inch))

        # RECOMMENDATION
        elements.append(Paragraph("CLINICAL RECOMMENDATION", style_h2))
        if grade <= 1:
            rec = "No significant arthritis. Routine follow-up."
        elif grade == 2:
            rec = "Mild arthritis detected. Consider physical therapy and lifestyle modifications."
        elif grade == 3:
            rec = "Moderate arthritis. Specialist referral recommended. Consider imaging follow-up in 6 months."
        else:
            rec = "Severe arthritis. Urgent orthopedic referral. Surgical consultation may be warranted."
        elements.append(Paragraph(rec, style_normal))

        # BUILD
        build_pdf(doc, elements)
        return filepath
    except Exception as e:
        logger.error(f"Error generating arthritis report: {e}")
        return None

def generate_osteoporosis_report(scan, patient, output_dir="reports/"):
    try:
        os.makedirs(output_dir, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        patient_name_safe = "".join([c if c.isalnum() else "_" for c in patient.get("full_name", "Unknown")])
        scan_id = getattr(scan, "id", scan.get("id", "unknown")) if not isinstance(scan, dict) else scan.get("id", "unknown")
        filename = f"osteoporosis_{scan_id}_{patient_name_safe}_{date_str}.pd"
        filepath = os.path.join(output_dir, filename).replace("\\\\", "/")
        
        report_id = str(uuid.uuid4())[:8].upper()
        report_date = datetime.now().strftime("%Y-%m-%d")
        report_time = datetime.now().strftime("%H:%M:%S")

        doc = create_report(filepath, report_id, report_date, report_time)
        elements = []

        styles = getSampleStyleSheet()
        style_h2 = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=12, spaceAfter=8, textTransform="uppercase", textColor=colors.HexColor("#333333"))
        style_normal = styles["Normal"]
        style_normal.fontName = "Helvetica"

        # TITLE
        elements.append(Paragraph("<b>MediScan AI &mdash; Osteoporosis Screening Report</b>", styles["Title"]))
        elements.append(Spacer(1, 0.2 * inch))

        # PATIENT INFO
        elements.append(Paragraph("PATIENT INFORMATION", style_h2))
        patient_data = [
            [Paragraph(f"<b>Name:</b> {patient.get('full_name', 'N/A')}", style_normal), Paragraph(f"<b>Patient ID:</b> {patient.get('id', 'N/A')}", style_normal)],
            [Paragraph(f"<b>Age:</b> {patient.get('age', 'N/A')}", style_normal), Paragraph(f"<b>Gender:</b> {patient.get('gender', 'N/A')}", style_normal)],
        ]
        t_patient = Table(patient_data, colWidths=[3.6 * inch, 3.6 * inch])
        t_patient.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP")]))
        elements.append(t_patient)
        elements.append(Spacer(1, 0.2 * inch))
        
        # DIAGNOSIS
        elements.append(Paragraph("DIAGNOSIS SECTION", style_h2))
        has_osteo = getattr(scan, "has_osteoporosis", False)
        badge_text = "<font color='white'><b>OSTEOPOROSIS DETECTED</b></font>" if has_osteo else "<font color='white'><b>NORMAL</b></font>"
        badge_bg = "#D32F2F" if has_osteo else "#388E3C"
        
        badge = Table([[Paragraph(badge_text, ParagraphStyle("Centered", parent=style_normal, alignment=TA_CENTER))]],
                      colWidths=[3.0 * inch], rowHeights=[0.3 * inch],
                      style=[("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(badge_bg)), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")])
        elements.append(badge)
        elements.append(Spacer(1, 0.1 * inch))
        
        conf = getattr(scan, "confidence", 0.0)
        conf_flag = getattr(scan, "confidence_flag", "low")
        elements.append(Paragraph(f"<b>Confidence:</b> {conf:.1f}% ({conf_flag})", style_normal))
        elements.append(Spacer(1, 0.2 * inch))

        # IMAGES
        img_path = getattr(scan, "original_file_path", "")
        heatmap_path = getattr(scan, "heatmap_path", "")
        img_table = Table([[get_image_flowable(img_path, "Original X-Ray", style_normal), get_image_flowable(heatmap_path, "Grad-CAM Heatmap", style_normal)]])
        elements.append(img_table)
        elements.append(Spacer(1, 0.2 * inch))
        
        # NOTE
        elements.append(Paragraph("<i>Note: Osteoporosis screening via X-ray is a preliminary assessment. DEXA scan is the gold standard for bone mineral density measurement.</i>", style_normal))
        elements.append(Spacer(1, 0.2 * inch))

        # OVERRIDE
        override = getattr(scan, "clinician_override", None)
        if override:
            elements.append(Paragraph("CLINICIAN OVERRIDE", style_h2))
            elements.append(Paragraph(f"<b>Override Diagnosis:</b> {override}", style_normal))
            elements.append(Paragraph(f"<b>Notes:</b> {getattr(scan, 'override_notes', 'N/A')}", style_normal))
            elements.append(Spacer(1, 0.2 * inch))

        # RECOMMENDATION
        elements.append(Paragraph("CLINICAL RECOMMENDATION", style_h2))
        if has_osteo and conf_flag == "clear":
            rec = "Positive screening. DEXA scan recommended for confirmation. Consider calcium/vitamin D supplementation and fall prevention counseling."
        elif has_osteo:
            rec = "Possible osteoporosis. DEXA scan strongly recommended for definitive diagnosis."
        else:
            rec = "No radiographic signs of osteoporosis. Routine screening per age-appropriate guidelines."
        elements.append(Paragraph(rec, style_normal))

        # BUILD
        build_pdf(doc, elements)
        return filepath
    except Exception as e:
        logger.error(f"Error generating osteoporosis report: {e}")
        return None
"""

with open("d:/X-ray ML Model/report_generator.py", "a") as f:
    f.write("\n" + report_code + "\n")
