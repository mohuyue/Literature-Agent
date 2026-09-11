"""读取文献PDF，提取正文文本、表格、图（按图注定位并渲染），构建向量索引，供 RAG 检索使用。"""
import re, os, json, base64, time, hashlib
import pymupdf, pdfplumber
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from config import (
    LLM_API_KEY,
    LLM_BASE_URL,
    VISION_MODEL,
    VISION_TIMEOUT,
    VISION_MAX_WORKERS,
    VISION_MAX_TOKENS,
    VISION_DISABLE_THINKING,
)

# 图注行：行首为 Fig. 5 / Figure 5 / Fig.5
_CAPTION_RE = re.compile(r'^\s*(?:Figure|Fig\.?)\s*(\d+)\b', re.I)
# 读取文本
def load_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    return text

# 提取表格
def extract_tables_as_markdown(pdf_path):
    """ 从 PDF 中提取所有表格，返回 Markdown 格式的字符串列表。"""
    tables_md = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table_num, table in enumerate(tables):
                if not table or len(table) < 2:
                    continue  # 跳过空表格或只有标题行的

                # 转成 Markdown 表格
                header = table[0]
                rows = table[1:]

                # 构建 Markdown 表格语法
                md_lines = []
                md_lines.append("| " + " | ".join(str(c) if c else "" for c in header) + " |")
                md_lines.append("| " + " | ".join("---" for _ in header) + " |")
                for row in rows:
                    md_lines.append("| " + " | ".join(str(c) if c else "" for c in row) + " |")

                # 加页码和表格编号标记，方便溯源
                table_md = f"【Table on Page {page_num+1}, Table {table_num+1}】\n" + "\n".join(md_lines)
                tables_md.append(table_md)

    return tables_md

# 提取图（按图注定位 + 渲染区域）
def _page_graphic_rects(page):
    """收集本页图形元素的矩形（矢量绘图 + 位图），用于确定图的真实区域。"""
    rects = []
    try:
        for d in page.get_drawings():
            r = d.get("rect")
            if r is not None and r.width > 1 and r.height > 1:
                rects.append(r)
    except Exception:
        pass
    try:
        for img in page.get_images(full=True):
            try:
                for r in page.get_image_rects(img[0]):
                    if r.width > 1 and r.height > 1:
                        rects.append(r)
            except Exception:
                continue
    except Exception:
        pass
    return rects


def extract_figures_from_pdf(pdf_path, dpi=150, min_height=40):
    """
    按【图注】定位并渲染每张图，返回列表：
    [{"figure_no":5,"page":5,"caption":"Fig. 5 ...","png_base64":...,"ext":"png","description":None}]

    为什么不用 page.get_images()：
      1) 只返回内嵌位图，矢量图（流程图/结构图）完全取不到；
      2) 一张图常由多张位图拼成，会被碎成多块；
      3) 返回顺序与论文的 Figure 编号无关，无法对应「图5」。

    做法：以「图注行（Fig. N）」为锚点，用其上方【图形元素的外接框】确定图区并渲染。
    这样既拿到真实图号，又能覆盖矢量图，还不受图内坐标标签干扰。
    """
    doc = pymupdf.open(pdf_path)
    candidates = []  # 每个候选：(fig_no, page, clip, caption, has_graphics)

    for page_num in range(len(doc)):
        page = doc[page_num]

        # 收集本页所有文本行 (y0, y1, text)
        lines = []
        for blk in page.get_text("dict").get("blocks", []):
            for ln in blk.get("lines", []):
                txt = "".join(s.get("text", "") for s in ln.get("spans", []))
                if txt.strip():
                    lines.append([ln["bbox"][1], ln["bbox"][3], txt])
        lines.sort(key=lambda x: x[0])

        # 图注行（行首为 Fig. N / Figure N）
        captions = []
        for i, (y0, y1, txt) in enumerate(lines):
            m = _CAPTION_RE.match(txt)
            if not m:
                continue
            cap_lines = [txt.strip()]
            for j in range(i + 1, min(i + 4, len(lines))):
                ny0, _ny1, ntxt = lines[j]
                if _CAPTION_RE.match(ntxt) or ny0 - y1 > 40:
                    break
                cap_lines.append(ntxt.strip())
            captions.append({
                "figure_no": int(m.group(1)),
                "y0": y0,
                "caption": " ".join(cap_lines),
                "label_words": len(txt.split()),  # 图注行词数：真图注通常很短，便于过滤正文里的 "Fig. N ..." 句子
            })

        if not captions:
            continue

        grects = _page_graphic_rects(page)
        prev_caption_y = 0.0
        for cap in captions:
            y0 = cap["y0"]
            window_top = max(y0 - 470, prev_caption_y + 2)
            window_bottom = y0 - 2
            in_window = [r for r in grects if r.y1 <= window_bottom and r.y0 >= window_top]

            top = bottom = g_bottom = None
            has_graphics = False
            if in_window:
                gx0 = min(r.x0 for r in in_window)
                gx1 = max(r.x1 for r in in_window)
                gtop = min(r.y0 for r in in_window)
                gbot = max(r.y1 for r in in_window)
                if gbot - gtop >= min_height:  # 外接框过薄（如一条分隔线）时不可信
                    x0, x1, top, bottom = gx0, gx1, gtop, gbot
                    g_bottom = gbot
                    has_graphics = True

            if not has_graphics:
                # 兜底：取图注上方最大空白带的下沿作为图区上界
                x0, x1 = page.rect.x0, page.rect.x1
                top, bottom = window_top, window_bottom
                cands = sorted(
                    [(ly0, ly1) for (ly0, ly1, _t) in lines
                     if window_top <= ly1 <= window_bottom],
                    key=lambda x: x[1],
                )
                best_gap, prev = 0.0, None
                for ly0, ly1 in cands:
                    if prev is not None and ly0 - prev > best_gap:
                        best_gap, top = ly0 - prev, ly0
                    prev = ly1

            # 图注与图之间是否夹着正文行：真图注正上方应是图，而不是正文文字
            between = 0
            if has_graphics:
                for ly0, ly1, _t in lines:
                    if ly1 <= y0 - 1 and ly0 >= g_bottom - 1:
                        between += 1

            clip = pymupdf.Rect(
                max(page.rect.x0, x0 - 5), max(page.rect.y0, top - 5),
                min(page.rect.x1, x1 + 5), min(page.rect.y1, bottom + 5),
            )
            prev_caption_y = y0
            if clip.height < min_height:
                continue
            candidates.append({
                "figure_no": cap["figure_no"],
                "page": page_num + 1,
                "clip": clip,
                "caption": cap["caption"],
                "has_graphics": has_graphics,
                "between": between,
                "short_label": cap["label_words"] <= 10,
            })

    # 按图号全局去重：上方确有图形 → 图注紧贴图（中间无正文） → 图注行短 → 区域大
    # （过滤正文里 "Fig. 9 displays the parity plots..." 这类以图号开头的句子）
    best = {}
    for c in candidates:
        key = c["figure_no"]
        score = (
            1 if c["has_graphics"] else 0,
            1 if c["between"] == 0 else 0,
            1 if c["short_label"] else 0,
            c["clip"].width * c["clip"].height,
        )
        cur = best.get(key)
        if cur is None or score > cur[0]:
            best[key] = (score, c)

    figures = []
    found_nos = []
    for fig_no in sorted(best.keys()):
        c = best[fig_no][1]
        try:
            png = doc[c["page"] - 1].get_pixmap(clip=c["clip"], dpi=dpi).tobytes("png")
        except Exception as e:
            print(f"[figures] Page {c['page']} Fig.{fig_no} 渲染失败：{e}", flush=True)
            continue
        figures.append({
            "figure_no": fig_no,
            "page": c["page"],
            "caption": c["caption"],
            "png_base64": base64.b64encode(png).decode("utf-8"),
            "ext": "png",
            "description": None,  # 后续由视觉模型填充
        })
        found_nos.append(fig_no)

    print(f"[figures] 定位到图：{sorted(found_nos)}", flush=True)
    return figures

# 用多模态LLM对图进行描述
vision_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=VISION_TIMEOUT, max_retries=0)
_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".vision_cache")


def _cache_key(pdf_path, fig):
    """描述缓存键：图片内容哈希 + 图号 + 模型。裁剪区域变化 → 图变 → 缓存自动失效。"""
    img_hash = hashlib.md5(fig["png_base64"].encode("utf-8")).hexdigest()
    raw = f"{os.path.abspath(pdf_path)}|{fig['page']}|{fig['figure_no']}|{VISION_MODEL}|{img_hash}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _cache_load(key):
    path = os.path.join(_CACHE_DIR, key + ".json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("description")
    except Exception:
        return None


def _cache_save(key, description):
    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(os.path.join(_CACHE_DIR, key + ".json"), "w", encoding="utf-8") as f:
        json.dump({"description": description}, f, ensure_ascii=False)


def describe_figures_with_vision(figures, pdf_path=None, max_workers=None):
    """
    用多模态 LLM 为每张图生成文字描述（并发，带磁盘缓存）。
    同一篇论文重复打开时命中缓存，秒级完成。
    """
    max_workers = max_workers or VISION_MAX_WORKERS
    total = len(figures)
    if total == 0:
        return figures

    print(f"[vision] 开始为 {total} 张图生成描述（并发 {max_workers} 路，模型 {VISION_MODEL}）...", flush=True)
    start = time.time()

    def work(fig):
        key = _cache_key(pdf_path, fig) if pdf_path else None
        if key:
            cached = _cache_load(key)
            if cached:
                fig["description"] = cached
                fig["from_cache"] = True
                return fig
        _describe_one_figure(fig)
        if key and fig.get("description"):
            _cache_save(key, fig["description"])
        return fig

    done = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for fig in ex.map(work, figures):
            done += 1
            tag = "缓存" if fig.get("from_cache") else "完成"
            print(f"[vision] {done}/{total} {tag}（Figure {fig['figure_no']}, Page {fig['page']}）", flush=True)
    print(f"[vision] 全部完成，用时 {time.time() - start:.1f}s", flush=True)

    return figures


def _describe_one_figure(fig):
    """为单张图生成描述（供并发调用）。"""
    prompt = (
        f"这是学术论文中的一张图。图注为：{fig['caption']}\n"
        "请用中文回答：\n"
        "1) 图表类型；\n"
        "2) 展示的主要数据 / 趋势 / 关系；\n"
        "3) 关键结论。\n"
        "只描述图中实际可见的内容，不要推测；直接给出描述，不要复述思考过程。"
    )
    try:
        call_kwargs = dict(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{fig['ext']};base64,{fig['png_base64']}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=VISION_MAX_TOKENS,
            temperature=0,
        )
        if VISION_DISABLE_THINKING:
            # 关闭推理模型的思考链：更快，且避免 content 被 reasoning 挤空
            try:
                response = vision_client.chat.completions.create(reasoning_effort="none", **call_kwargs)
            except TypeError:
                # 个别 SDK 版本不认 reasoning_effort，降级为普通调用
                response = vision_client.chat.completions.create(**call_kwargs)
        else:
            response = vision_client.chat.completions.create(**call_kwargs)

        msg = response.choices[0].message
        text = (msg.content or "").strip()
        if not text:
            # 兜底：推理型模型可能把内容放进 reasoning_content，而 message.content 为空
            text = (getattr(msg, "reasoning_content", None) or "").strip()
        fig["description"] = text or f"【Figure {fig['figure_no']} 未能生成描述】"

    except Exception as e:
        # 调用失败/超时 —— 降级处理，并打印真实原因（不静默）
        print(f"[vision] Figure {fig['figure_no']} (Page {fig['page']}) 失败：{type(e).__name__}: {e}", flush=True)
        fig["description"] = f"【Figure {fig['figure_no']} (Page {fig['page']}) 视觉描述失败】"
    return fig

# ==================== 使用示例（可选择性添加） ====================
if __name__ == "__main__":
    pdf_file = "Zheng 等 - 2025 - Machine Learning‐Driven Prediction of Polypropylene Alloy Properties A Bayesian Network– span styl.pdf"
    

    with pdfplumber.open(pdf_file) as pdf:

    #     for page_num, page in enumerate(pdf.pages):

    #         tables = page.extract_tables({
    #             "vertical_strategy": "lines",
    #             "horizontal_strategy": "lines"
    #         })

    #         print(
    #             f"Page {page_num+1}: "
    #             f"{len(tables)} tables"
    #         )
        
        for page in pdf.pages:

            text = page.extract_text()

            if text and "ALGORITHM" in text:
                print(text[:1000])


    
    # 提取表格（Markdown 格式）
    # tables = extract_tables_as_markdown(pdf_file)
    # print(f"提取表格数量：{len(tables)}")
    # for i, table in enumerate(tables[:-1]):
    #     print("\n" + "="*80)
    #     print(f"Table {i+1}")
    #     print("="*80)
    #     print(table)
    
    # # 提取图片并生成描述
    # images = extract_images_from_pdf(pdf_file)
    # print(f"提取图片数量（过滤后）：{len(images)}")
    
    # 调用多模态生成描述（需确保 config 中配置了有效的模型）
    # if images:
    #     described_images = describe_images_with_vision(images)
    #     for img in described_images:
    #         print(f"Page {img['page']}, Image {img['image_num']}: {img['description'][:80]}...")