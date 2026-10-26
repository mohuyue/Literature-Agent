"""Agent 侧的模型调用客户端。

关键：本模块**不 import xgboost / shap / sklearn 的任何重型对象**，只做
「读 metadata.json」和「起子进程调 worker」。这样 Agent 环境（numpy 2.4.6 /
sklearn 1.9.0）与模型训练环境（numpy 1.26.4 / sklearn 1.5.1 / xgboost 2.1.0）
可以完全解耦。
"""
import json
import os
import subprocess
import sys

from config import ML_PYTHON, ML_MODELS_DIR, ML_WORKER_TIMEOUT

# worker.py 的绝对路径：与本文件同目录，不依赖调用方的 cwd
WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")

# metadata 进程内缓存：{model_id: dict}。只在导出模型时变化，运行期不会改。
_METADATA_CACHE = {}


class MLError(RuntimeError):
    """模型侧可预期的失败（模型不存在、参数不合法、推理超时、worker 崩溃）。

    单独定义异常类型，是为了让 tools.py 能"接住并转成给 LLM 看的文本"，
    而不是让整个对话因为一次模型调用失败而中断。
    """
    pass


def model_ids():
    """列出可用模型 ID。

    判定标准是"目录里有 metadata.json"，而不是"目录存在"：
    模型必须带说明书才能被 Agent 正确调用，缺说明书的目录等于不可用。
    """
    if not os.path.isdir(ML_MODELS_DIR):
        return []
    return [
        name for name in sorted(os.listdir(ML_MODELS_DIR))
        if os.path.isfile(os.path.join(ML_MODELS_DIR, name, "metadata.json"))
    ]


def _norm_id(value):
    """把模型 ID 归一化成"只保留字母数字"的形式，用于宽松比对。

    目录名里带空格、连字符和大写下划线混用（如 'BNXGB_BN-guided optimized'），
    LLM 转述时很容易少一个空格或改大小写。严格比对会让这种小失误变成
    "模型不存在"，而它其实是同一个模型。
    """
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


def _resolve_model_id(model_id):
    """返回真实存在的模型 ID；找不到返回 None。

    先精确匹配，再归一化匹配。不做模糊/相似度匹配 —— 三个模型名很像，
    猜错的代价是"用错一代模型还给出解释"，比直接报错严重得多。
    """
    ids = model_ids()
    if model_id in ids:
        return model_id
    target = _norm_id(model_id)
    for cand in ids:
        if _norm_id(cand) == target:
            return cand
    return None


def get_metadata(model_id):
    """读某个模型的 metadata.json（带进程内缓存）。

    这是 Agent 侧唯一的信息来源：输入规范、单位、范围、精度全从这里来。
    缓存是安全的——metadata 只在导出时变化，运行中不会改；
    改了模型需要重启 Streamlit 才能生效，这与"重载模型"的预期一致。
    """
    if model_id in _METADATA_CACHE:
        return _METADATA_CACHE[model_id]
    resolved = _resolve_model_id(model_id) or model_id
    path = os.path.join(ML_MODELS_DIR, resolved, "metadata.json")
    if not os.path.isfile(path):
        raise MLError("模型 '{}' 不存在。可用模型：{}".format(
            model_id, "、".join(model_ids()) or "（无）"))
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    _METADATA_CACHE[model_id] = meta
    _METADATA_CACHE[resolved] = meta
    return meta


def resolve_model_id(model_id):
    """公开的解析入口，供 client.call_worker 在起子进程前校正 ID。"""
    return _resolve_model_id(model_id) or model_id


def list_models():
    """返回精简的模型清单（不含大段 schema）。

    只抽取"选模型所需的字段"：能预测什么、要几个输入。
    完整 schema 由 get_model_input_schema 单独返回，
    避免第一次列清单就把几千字塞进上下文。
    """
    out = []
    for model_id in model_ids():
        try:
            meta = get_metadata(model_id)
        except MLError:
            continue
        targets = meta.get("targets", [])
        viscosity_rules = (meta.get("viscosity_conversion") or {}).get("rules", [])
        family = meta.get("family") or {}
        out.append({
            "model_id": model_id,
            "display_name": meta.get("display_name", model_id),
            # description 必须回传：三个模型输入成本差很多（4 / 10 / 14 个），
            # Agent 要靠它判断"用户手上有什么数据时该用哪个"，不能只按名字猜。
            "description": meta.get("description", ""),
            "task": meta.get("task"),
            "targets": ["{}({})".format(t.get("label", t["name"]), t.get("unit", "-")) for t in targets],
            "n_required_inputs": len(meta.get("inputs", {}).get("required", [])),
            "n_optional_inputs": len(meta.get("inputs", {}).get("optional", [])),
            # 有多少个分子量输入可以用特性粘度替代 —— 决定 Agent 要不要主动问 η
            "n_viscosity_alternatives": len(viscosity_rules),
            # 代际信息：三个模型是同一路线的 1→2→3 代，不是并列候选。
            # 没有这段，Agent 面对"能预测同样 4 个指标"的三个模型无法解释该选谁。
            "stage": family.get("stage"),
            "stage_name": family.get("stage_name"),
            "stage_summary": family.get("stage_summary"),
            "family_name": family.get("name"),
            "trained_on": meta.get("trained_on", {}),
        })
    # 按代际排序：让"1→2→3"的顺序直接体现在给 LLM 的文本里，而不是靠它自己排
    out.sort(key=lambda m: (m["stage"] is None, m["stage"] or 0))
    return out


def call_worker(payload):
    """起一个 worker 子进程，返回其 JSON 结果。

    这是"跨解释器调用"的唯一入口，也是整个方案的技术核心：
        [Agent 环境 python] --stdin JSON--> [训练环境 python + worker.py]
                            <--stdout JSON--

    三个必须注意的点：
      1) 用 ML_PYTHON 指定的解释器，而不是 sys.executable。
         Agent 环境没有 xgboost/shap，用它启动必然 ImportError。
      2) stdout 里可能混入第三方库的 warning，所以不能直接 json.loads(整个输出)，
         要反向找第一行能解析的 JSON（见下）。
      3) stderr 原样转发到当前进程的 stderr —— 只做可观测性，不参与结果解析。
    """
    if not os.path.isfile(ML_PYTHON):
        raise MLError(
            "找不到模型推理解释器：{}\n"
            "请在 .env 中设置 ML_PYTHON 指向训练环境（需含 xgboost/shap/sklearn）。".format(ML_PYTHON)
        )

    # 先校正模型 ID：worker 用目录名拼路径，少一个空格就会 FileNotFound。
    # 这里统一成真实目录名，下游（返回结果、图片命名）也就都一致了。
    if payload.get("model_id"):
        payload = dict(payload, model_id=resolve_model_id(payload["model_id"]))

    try:
        proc = subprocess.run(
            [ML_PYTHON, WORKER_PATH],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            # 训练环境输出里可能有 GBK 片段，用 replace 兜底，避免 UnicodeDecodeError 把整次调用打断
            errors="replace",
            timeout=ML_WORKER_TIMEOUT,
            # worker 里用的是相对项目根的路径（models/、ML_model/），
            # 这里固定 cwd 就不必依赖调用方当前目录
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
    except subprocess.TimeoutExpired:
        raise MLError("模型推理超时（>{}s）".format(int(ML_WORKER_TIMEOUT)))

    if proc.stderr.strip():
        # 转发而不是丢弃：Vision/xgboost 的告警、SHAP 画图失败都在这里
        print("[ml-worker] " + proc.stderr.strip().replace("\n", " | "), file=sys.stderr, flush=True)

    # 只取最后一行可解析的 JSON —— 库的告警可能混进 stdout
    for line in reversed((proc.stdout or "").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            result = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not result.get("ok", False):
            raise MLError(result.get("error", "模型推理失败"))
        return result.get("data", {})

    # 走到这里说明 worker 没给出结构化结果：把退出码和 stderr 一起抛出，
    # 否则上层只能看到一句"失败了"，无法定位是导入错误还是路径错误。
    raise MLError(
        "模型推理进程没有返回有效结果（exit={}）。stderr：{}".format(
            proc.returncode, (proc.stderr or "").strip()[:500] or "（空）"
        )
    )


def predict(model_id, values, targets=None):
    """预测。targets 为空表示预测该模型的全部目标。"""
    payload = {"command": "predict", "model_id": model_id, "values": values}
    if targets:
        payload["targets"] = targets
    return call_worker(payload)


def explain(model_id, values, target, top_k=8):
    """SHAP 解释。values 必须与 predict 时完全一致，否则解释的不是同一次预测。

    top_k 只影响返回的贡献列表长度；图里始终画全部父节点（否则看不清整体结构）。
    """
    return call_worker({
        "command": "explain",
        "model_id": model_id,
        "values": values,
        "target": target,
        "top_k": top_k,
    })


def describe(model_id):
    """返回节点图与拓扑序，排障用（当前没有对应的 Agent 工具）。"""
    return call_worker({"command": "describe", "model_id": model_id})
