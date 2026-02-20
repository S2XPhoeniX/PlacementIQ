
import pdfplumber

def extract_text_from_pdf(uploaded_file):
    try:
        text_pages = []

        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    text_pages.append(txt)

        return True, "\n".join(text_pages)

    except Exception:
        return False, ""