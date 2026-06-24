"""清洗文本，去掉论文参考文献 doi URL """
import re

# def clean_text(text):
    
#     # 去参考文献
#     idx = text.lower().find("references")
#     if idx != -1:
#         text = text[:idx]

#     # 去URL/DOI
#     text = re.sub(r"http\S+", "", text)
#     text = re.sub(r"doi:\S+", "", text)
    
#     # 去多余空格
#     text = re.sub(r"\s+", " ", text)

#     return text

def clean_text(text):

    # 找最后一个 "references" 而非第一个，避免截断正文
    lower = text.lower()
    idx = lower.rfind("\nreferences")   # 用 \n 开头限定，要求它是独立行
    if idx == -1:
        idx = lower.rfind("references")  # 降级：找不到带换行的就找普通的最后一个
    if idx != -1:
        text = text[:idx]

    # 去URL/DOI
    import re
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"doi:\S+", "", text)

    # 去多余空格
    text = re.sub(r"\s+", " ", text)

    return text