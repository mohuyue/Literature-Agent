"""清洗文本，去掉论文参考文献 doi URL """
import re

def clean_text(text):
    
    # 去参考文献
    idx = text.lower().find("references")
    if idx != -1:
        text = text[:idx]

    # 去URL/DOI
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"doi:\S+", "", text)
    
    # 去多余空格
    text = re.sub(r"\s+", " ", text)

    return text