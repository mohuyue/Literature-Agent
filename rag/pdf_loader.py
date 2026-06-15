"""读取文献PDF，提取可复制文本（不包含图片）"""
import pymupdf

def load_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    return text