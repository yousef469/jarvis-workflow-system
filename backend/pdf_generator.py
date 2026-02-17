from fpdf import FPDF
import datetime
import os

class JarvisPDF(FPDF):
    def header(self):
        self.set_font('helvetica', 'B', 15)
        self.cell(0, 10, 'J.A.R.V.I.S. MEDICAL REPORT', border=False, align='C')
        self.ln(10)
        self.set_line_width(1)
        self.line(10, 22, 200, 22)
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}} - Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}', align='C')

def clean_text(text):
    """
    Cleans text for FPDF (standard fonts only support Latin-1).
    Replaces common curly quotes and non-latin characters.
    """
    replacements = {
        '\u2018': "'", '\u2019': "'", # Smart quotes
        '\u201c': '"', '\u201d': '"', # Smart double quotes
        '\u2013': '-', '\u2014': '-', # En/Em dashes
        '\u2022': '*',                # Bullet points
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Final fallback: encode to latin-1 and back, ignoring errors
    try:
        return text.encode('latin-1', 'replace').decode('latin-1')
    except:
        return text

def create_medical_pdf(filename, topic, content):
    """
    Creates a professional medical PDF based on LLM content.
    """
    print(f"[PDF-Engine] Starting generation: {filename}")
    try:
        pdf = JarvisPDF()
        pdf.alias_nb_pages()
        pdf.add_page()
        
        # Title Section
        pdf.set_font('helvetica', 'B', 14)
        pdf.cell(0, 10, f'Subject: {clean_text(topic).upper()}', ln=True)
        pdf.set_font('helvetica', '', 12)
        pdf.cell(0, 10, f'Requested by: Yousef', ln=True)
        pdf.ln(5)
        
        # Content Section
        pdf.set_font('helvetica', '', 11)
        # Handle multi-line content
        lines = content.split('\n')
        for line in lines:
            cleaned_line = clean_text(line.strip())
            if cleaned_line:
                pdf.multi_cell(0, 8, cleaned_line)
                pdf.ln(2)
            else:
                pdf.ln(4)
                
        # Save file
        output_path = os.path.abspath(filename)
        pdf.output(output_path)
        print(f"[PDF-Engine] Successfully saved to: {output_path}")
        
        # Auto-index in Memory & Ingest in Document Engine
        try:
            from memory_engine_v2 import memory_v2
            
            memory_v2.add_asset(
                type="pdf",
                path=output_path,
                description=f"Generated PDF report on {topic}"
            )
        except Exception as e:
            print(f"[PDF-Engine] Indexing/Ingestion error: {e}")

        return output_path
    except Exception as e:
        print(f"[PDF-Engine] CRITICAL FAILURE: {e}")
        return None

if __name__ == "__main__":
    # Test
    create_medical_pdf("test_report.pdf", "ACL Injury", "Sample medical details for doctor.")
