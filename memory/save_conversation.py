"""把一轮对话（含图片）落盘到 memory/ 下的 Markdown 与图片目录。

为什么要"拷贝图片"而不是在 md 里写中转目录的原始路径：
    SHAP 图由 ml/worker.py 在子进程里画到系统临时目录（tempfile.gettempdir()），
    它只是"子进程出图 → 父进程取用"的一次性中转站，文件名带随机 uuid，
    会被系统清理；而 memory/ 是长期记忆。
    把图画进长期记忆里，就必须先复制到一个不会被清理的位置，
    否则第二天打开 md 只剩一个裂图链接。归档完成后 app.py 会删掉中转文件，
    所以磁盘上不会长期存在两份。

目录约定：
    memory/2026-10-25.md                 当天的对话记录（按轮追加）
    memory/images/2026-10-25/xxx.png     当天的图片
md 里用相对路径 images/2026-10-25/xxx.png —— 相对于 md 文件自身，
这样整个 memory/ 目录可以整体移动/备份，链接不会失效。
"""
import os
import shutil
from datetime import datetime

# 本文件就在 memory/ 下，用 __file__ 定位，避免依赖调用方的当前工作目录
MEMORY_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR_NAME = "images"


def _safe_filename(name):
    """把模型名里的空格等字符换成下划线，得到可直接放进 Markdown 链接的文件名。

    为什么需要：SHAP 图的文件名由 model_id 拼成，而 model_id 可能含空格
    （例如 "BNXGB_BN-guided optimized_IS_xxxx.png"）。Markdown 的图片语法
    ![](path with space) 在多数渲染器里会在空格处断开，图就成了死链。
    """
    out = []
    for ch in name:
        if ch.isalnum() or ch in "._-":
            out.append(ch)
        else:
            out.append("_")
    return "".join(out) or "image.png"


def persist_image(src_path, day=None):
    """把一张图片复制到 memory/images/<日期>/，返回 (绝对路径, md 里的相对路径)。

    调用方（app.py）需要绝对路径来 st.image()，md 里需要相对路径。
    两个都返回，免得调用方自己拼路径、拼错斜杠方向。

    拷贝失败不抛异常：图片存不进长期记忆，不该让整轮对话失败。
    返回 (None, None) 由调用方决定如何提示。
    """
    if not src_path or not os.path.isfile(src_path):
        return None, None

    day = day or datetime.now().strftime("%Y-%m-%d")
    target_dir = os.path.join(MEMORY_DIR, IMAGE_DIR_NAME, day)
    os.makedirs(target_dir, exist_ok=True)

    # 保留原文件名（含 model_id / target / uuid）便于辨认，只把不安全字符换成下划线；
    # uuid 已保证不重名，不需要额外加时间戳。
    filename = _safe_filename(os.path.basename(src_path))
    dst = os.path.join(target_dir, filename)
    try:
        if os.path.abspath(src_path) != os.path.abspath(dst):
            shutil.copy2(src_path, dst)
    except OSError:
        return None, None

    rel = "{}/{}/{}".format(IMAGE_DIR_NAME, day, filename)
    return dst, rel


def image_caption(image):
    """给图片生成一句图注：优先用"目标中文名（模型）"，退化到文件名。"""
    if not isinstance(image, dict):
        return "SHAP 解释（瀑布图）"
    label = image.get("target_label") or image.get("target") or "预测"
    model_id = image.get("model_id") or ""
    unit = image.get("unit") or ""
    cap = "SHAP 解释：{}{}".format(label, "（{}）".format(unit) if unit else "")
    if model_id:
        cap += " — 模型 {}".format(model_id)
    return cap


def save_conversation(user_input, answer, images=None):
    """追加一轮问答到 memory/<日期>.md。

    参数:
        user_input: 用户问题
        answer:     助手最终回答（纯文本）
        images:     本轮要归档的图片列表，元素可以是
                      - dict：{"path": 绝对路径, "md_rel": 相对路径, "target_label": ..., "model_id": ...}
                      - str：图片绝对路径（兼容旧调用）
                    path 为空则跳过。

    为什么图片信息要在 md 里单独成行、并带图注：
    只写路径的 md 打开后只有一行字；写成 Markdown 图片语法，
    支持图片的编辑器（Typora / VSCode / Obsidian）会直接渲染出瀑布图，
    这一点对"回看当初为什么这么解释"很关键。
    """
    os.makedirs(MEMORY_DIR, exist_ok=True)

    now = datetime.now()
    day = now.strftime("%Y-%m-%d")
    md_file = os.path.join(MEMORY_DIR, "{}.md".format(day))

    lines = [
        "",
        "# {}".format(now.strftime("%H:%M:%S")),
        "",
        "## User",
        user_input,
        "",
        "## Assistant",
        answer,
        "",
    ]

    for image in images or []:
        if isinstance(image, dict):
            abs_path = image.get("path")
            rel_path = image.get("md_rel")
        else:
            abs_path = image
            rel_path = None

        if not abs_path:
            continue

        if not rel_path:
            # 没走过 persist_image（例如直接传了路径）→ 就地补一次归档，
            # 保证 md 里的链接一定指向 memory 内部的副本。
            abs_path, rel_path = persist_image(abs_path, day)
            if not rel_path:
                continue

        lines.append("![{}]({})".format(image_caption(image), rel_path))
        lines.append("")

    with open(md_file, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n---\n")
