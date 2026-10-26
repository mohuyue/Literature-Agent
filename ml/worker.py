"""模型推理 worker —— 必须在【训练环境】中运行。

    E:\\anaconda\\envs\\newenv\\python.exe ml\\worker.py

设计原因：Agent 所在环境（literature_agent：numpy 2.4.6 / sklearn 1.9.0）与训练环境
（newenv：numpy 1.26.4 / sklearn 1.5.1 / xgboost 2.1.0 / shap 0.46.0）不一致。
pkl 由训练环境序列化，跨版本加载可能静默改变行为；且 shap 依赖 numba（要求 numpy<2.3），
硬装会降级 numpy 并波及 torch / faiss / sentence-transformers。
所以 Agent 侧只通过 subprocess 传 JSON 调用本文件。

协议：stdin 读一个 JSON，stdout 写一个 JSON（最后一行），日志一律走 stderr。

  {"command": "describe", "model_id": "BNXGB_BN-guided optimized"}
  {"command": "predict",  "model_id": "...", "values": {"E_sol": 24, ...}}
  {"command": "explain",  "model_id": "...", "values": {...}, "target": "IS", "top_k": 8}

支持的两种模型结构（都由 metadata.json 的 kind 描述，但计算逻辑统一）：
  bn_node_graph  BN-XGB：18 个节点，有 root/formula/linear/xgb 四类，可自动推断缺失节点
  flat_xgb       纯 XGB：4 个目标，直接吃 10 或 14 个特征，无推断
"""
import json
import os
import pickle
import sys
import tempfile
import traceback
import uuid

# matplotlib 必须在导入 pyplot/shap 之前切到无界面后端
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "ML_model", "models")
MODELS_DIR = os.environ.get("ML_MODELS_DIR") or DEFAULT_MODELS_DIR

# SHAP 图的中转目录。
# 为什么必须落盘：explain 跑在独立的训练环境子进程里（那边才有 xgboost/shap/
# matplotlib），父进程拿不到进程内的 Figure 对象，stdout 只能传路径。
# 为什么不放项目目录：这里只是"子进程画图 → 父进程取用"的一次性中转站，
# 长期副本由 memory/save_conversation.persist_image() 归档到 memory/images/<日期>/，
# 归档完成后由 app.py 删除源文件。放系统临时目录，项目目录不会再囤积 PNG。
PLOT_DIR = os.path.join(tempfile.gettempdir(), "literature_agent_shap")

_CACHE = {}


# --------------------------------------------------------------------------
# 加载
# --------------------------------------------------------------------------
def model_dir(model_id):
    """models/<model_id>/ 的绝对路径。"""
    return os.path.join(MODELS_DIR, model_id)


def load_metadata(model_id):
    """读说明书。worker 只把 metadata 当成"输出用的元信息"，不参与计算决策。"""
    path = os.path.join(model_dir(model_id), "metadata.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_id):
    """反序列化 pkl，并按 model_id 缓存。

    缓存的意义：一次 explain 会反复用到同一个 pkl（5 个折各一次），
    做成函数内缓存避免重复读盘。进程本身是一次性的，所以不需要考虑失效。
    """
    if model_id in _CACHE:
        return _CACHE[model_id]
    meta = load_metadata(model_id)
    entry = meta.get("entry", {})
    pkl_path = os.path.join(model_dir(model_id), entry.get("file", "model.pkl"))
    with open(pkl_path, "rb") as f:
        model = pickle.load(f)
    # 目前只支持"dict 描述的模型"：BN 是 18 节点图，纯 XGB 是 4 个目标的扁平表。
    # 其他结构（例如 sklearn Pipeline）需要先在这里加一条分派，否则宁可报错也别乱猜。
    if not isinstance(model, dict):
        raise TypeError("不支持的模型结构：期望 {节点: {type, parents, ...}}")
    # 兼容性补齐：早期训练脚本产出的扁平模型只有 {parents, models, scalers}，
    # 没有 type 字段。结构上它就是"用 5 折 XGB 拟合目标"，补齐后
    # 计算、SHAP、拓扑排序都走同一条路径，不必按 model_id 写特例。
    for node, info in model.items():
        if isinstance(info, dict) and "type" not in info:
            if not info.get("models"):
                raise TypeError("节点 '{}' 缺少 type 且不含 models，无法判断类型".format(node))
            info["type"] = "xgb"
    _CACHE[model_id] = (model, meta)
    return model, meta


def list_model_ids():
    """扫描 models/ 下所有带 metadata.json 的子目录。

    只认 metadata.json 存在，避免把 __pycache__ 之类的非模型子目录误当成模型。
    """
    if not os.path.isdir(MODELS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(MODELS_DIR)):
        if os.path.isfile(os.path.join(MODELS_DIR, name, "metadata.json")):
            out.append(name)
    return out


# --------------------------------------------------------------------------
# 图结构
# --------------------------------------------------------------------------
def node_deps(info):
    """节点的全部依赖。必须取 parents ∪ inputs：

    公式节点的 inputs 可能包含非直接父节点（例如 E_var 用到 E_total，
    但 E_total 不在 E_var 的 parents 里）。原 notebook 只补 parents，
    只给根节点时会 KeyError —— 这里修掉。
    """
    deps = []
    for name in list(info.get("parents") or []) + list(info.get("inputs") or []):
        if name not in deps:
            deps.append(name)
    return deps


def topo_order(model):
    """Kahn 拓扑排序，返回"父节点一定排在子节点前面"的节点序列。

    为什么必须排序：BN 的计算是链式的（E_total → E_var → IS），
    随机按 dict 顺序算会在取 values[父节点] 时 KeyError。
    扁平模型（纯 XGB）没有节点内依赖，排序结果就是 4 个目标本身，
    调用方无需区分两种结构。
    """
    nodes = list(model.keys())
    deps = {n: [d for d in node_deps(model[n]) if d in model] for n in nodes}
    children = {n: [] for n in nodes}
    indeg = {}
    for n in nodes:
        indeg[n] = len(deps[n])
        for d in deps[n]:
            children[d].append(n)

    queue = [n for n in nodes if indeg[n] == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for child in children[node]:
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)

    if len(order) != len(nodes):
        raise ValueError("模型依赖图存在环，无法拓扑排序")
    return order


# --------------------------------------------------------------------------
# 单节点计算
# --------------------------------------------------------------------------
def _dmatrix(x, booster):
    """构造 XGBoost 的 DMatrix，并在必要时带上列名。

    为什么需要列名：第 1 代模型是在带列名的 DataFrame 上训练的，Booster 记住了
    feature_names；这时再喂一个没有列名的 DMatrix，xgboost 会直接报
    "training data did not have the following fields"。而第 2、3 代的 Booster
    没有记录列名，传不传都行。

    所以由 Booster 自己决定：它有列名就带上（名字必须是训练时的那些，
    即 parents 的顺序），没有就不带。硬编码列名会在三个模型之间串味。
    """
    import xgboost as xgb
    names = getattr(booster, "feature_names", None)
    if names:
        return xgb.DMatrix(x, feature_names=list(names))
    return xgb.DMatrix(x)


def compute_node(model, node, values):
    """按节点类型算出一个值。四类节点对应四种先验知识：

        root     没有父节点，用训练集均值（"不知道就取平均"）
        formula  领域公式，纯代数，不需要学习（比值 / 加权方差）
        linear   5 折线性回归取平均（中间量用简单模型，避免过拟合）
        xgb      5 折 XGBoost 取平均（目标节点用强模型）

    values 是"已经算好的变量字典"，本函数只负责当前这一个节点。
    """
    info = model[node]
    node_type = info["type"]

    # 根节点：pkl 里存了训练集均值。用户不提供时用它兜底，
    # 保证 BN 在信息不全时也能出一条结果（而不是崩掉）。
    if node_type == "root":
        return float(info["mean"])

    # 先验知识节点：公式在训练时就固定下来，推理时原样套用。
    if node_type == "formula":
        formula = info.get("formula_type")
        ins = info["inputs"]
        if formula == "ratio":
            return float(values[ins[0]] / values[ins[1]])
        if formula == "weighted_var":
            # 加权方差：以可溶物质量分数为权重，衡量两组分的差异程度
            mass = values[ins[0]] / 100.0
            return float(
                mass * (values[ins[1]] - values[ins[3]]) ** 2
                + (1.0 - mass) * (values[ins[2]] - values[ins[3]]) ** 2
            )
        raise ValueError("未知公式类型：{}".format(formula))

    # 学习型节点（linear / xgb）：父节点按 pkl 里记录的顺序取，顺序错了结果就错。
    parents = info["parents"]
    x = np.array([[values[p] for p in parents]], dtype=float)

    # 5 折模型逐个预测再平均 —— 与训练时的 OOF 口径、以及 notebook 的推理写法一致。
    # 注意必须先 scaler.transform：scaler 是每折单独拟合的，不能共用。
    preds = []
    for est, scaler in zip(info["models"], info["scalers"]):
        x_scaled = scaler.transform(x)
        if node_type == "xgb":
            preds.append(float(est.predict(_dmatrix(x_scaled, est))[0]))
        else:
            preds.append(float(est.predict(x_scaled)[0]))
    return float(np.mean(preds))


# --------------------------------------------------------------------------
# 校验与推断
# --------------------------------------------------------------------------
def _ranges(meta):
    ranges = {}
    for item in meta.get("inputs", {}).get("required", []):
        ranges[item["name"]] = item.get("range")
    for item in meta.get("inputs", {}).get("optional", []):
        ranges[item["name"]] = item.get("range")
    for item in meta.get("targets", []):
        ranges[item["name"]] = item.get("range")
    # 特性粘度也要参与范围校验：粘度超出训练范围时，换算出来的分子量同样属于外推
    for rule in (meta.get("viscosity_conversion") or {}).get("rules", []):
        ranges[rule["eta"]] = rule.get("eta_range")
    return ranges


# --------------------------------------------------------------------------
# 特性粘度 → 粘均分子量
# --------------------------------------------------------------------------
def viscosity_setup(meta):
    """从 metadata 读出换算参数与规则，返回 (参数, {目标变量: 规则})。

    为什么规则来自 metadata 而不是写死在代码里：
    K_PE / alpha 是领域常数（与聚合物种类、溶剂体系有关），换一个牌号就可能不同。
    放进说明书后，换模型只改 json；Agent 也能读到方程并如实转述给用户。
    """
    block = meta.get("viscosity_conversion") or {}
    if not block.get("enabled"):
        return None, {}
    params = {
        "K_PE": float(block.get("K_PE") or 4.06e-4),
        "alpha": float(block.get("alpha") or 0.725),
    }
    rules = {}
    for rule in block.get("rules") or []:
        if rule.get("eta") and rule.get("target"):
            rules[rule["target"]] = rule
    return params, rules


def eta_to_mv(eta, ethylene_wt, params):
    """修正 Mark-Houwink 方程：由特性粘度 [η] 与乙烯含量算粘均分子量。

        M = ( [η] / ( K_PE * (2/3 + 1/3 * W_E)^(alpha+1) ) )^(1/alpha)

    W_E 是乙烯单元的质量分数，而数据里的 E_* 是 wt%（0~35），所以先除以 100。
    (2/3 + 1/3·W_E) 是对共聚组成的修正：乙烯单元越多、链越柔顺，
    相同分子量下测得的特性粘度越低，不做修正会把分子量系统性算错。
    """
    if eta is None or eta <= 0:
        raise ValueError("特性粘度必须为正数，收到 {!r}".format(eta))

    w_e = float(ethylene_wt) / 100.0
    base = 2.0 / 3.0 + w_e / 3.0
    # 只有当 W_E < -2 时修正项才会非正，属于数据错误而不是外推，必须报错
    if base <= 0:
        raise ValueError(
            "乙烯含量 W_E={:.4g} 使修正项 (2/3 + 1/3·W_E) ≤ 0，无法换算分子量".format(w_e))

    denom = params["K_PE"] * base ** (params["alpha"] + 1.0)
    return float((eta / denom) ** (1.0 / params["alpha"]))


def validate(model, meta, values):
    """校验并清洗用户输入，返回 (清洗后的值, 错误列表, 告警列表)。

    分三档处理，区别很重要：
      - 错误：让本次调用失败，Agent 必须回头问用户（类型不对、必填缺失）
      - 告警：照常计算，但必须告知用户结果不可信（超出训练范围＝外推）
      - 通过：静默转成 float 后交给 build_values
    """
    errors, warnings = [], []
    required = [x["name"] for x in meta.get("inputs", {}).get("required", [])]
    optional = [x["name"] for x in meta.get("inputs", {}).get("optional", [])]

    # 变量别名归一化。第 1 代模型的特征名是希腊字母 η（'η_total'），
    # 而另外两代用 ASCII 的 'eta_total' 表示"特性粘度的替代输入"。
    # LLM 与用户都极可能写成 ASCII 版，若不归一化就会被判成"未知变量"。
    # 别名表放在 metadata 里，换模型只需改 json。
    alias = meta.get("input_aliases") or {}
    if alias:
        normalized = {}
        for key, val in values.items():
            canonical = alias.get(key, key)
            # 两个别名指向同一变量却给了不同数值 → 无法判断以谁为准，必须报错
            if canonical in normalized and normalized[canonical] != val:
                errors.append(
                    "变量 '{}' 与 '{}' 指向同一个输入但取值不同，请只提供一个".format(
                        key, canonical))
                continue
            normalized[canonical] = val
        values = normalized

    # 特性粘度是"分子量的替代输入"：它不在模型图里，也不在 required 里，
    # 但必须被当成合法变量接受，否则用户一给 eta_sol 就会被判成"未知变量"。
    params, rules = viscosity_setup(meta)
    eta_names = [r["eta"] for r in rules.values()]

    # 已知变量 = 模型节点 ∪ metadata 声明的输入 ∪ 特性粘度输入。
    # 为什么不能只用 model.keys()：
    #   BN 模型里输入本身就是图节点，keys() 够用；
    #   纯 XGB 是扁平结构，model 里只有 4 个目标键，
    #   14 个特征根本不在其中 —— 只查 keys() 会把正常输入全判成"未知变量"。
    known = set(model.keys()) | set(required) | set(optional) | set(eta_names)
    ranges = _ranges(meta)

    for name in values:
        if name not in known:
            errors.append("未知变量 '{}'。可用输入变量：{}".format(
                name, "、".join(required + optional)))

    cleaned = {}
    for name, raw in values.items():
        if name not in known:
            continue
        if isinstance(raw, bool) or raw is None:
            errors.append("变量 '{}' 的取值必须是数字，收到 {!r}".format(name, raw))
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            errors.append("变量 '{}' 的取值必须是数字，收到 {!r}".format(name, raw))
            continue
        if not np.isfinite(val):
            errors.append("变量 '{}' 取值不是有限数".format(name))
            continue
        cleaned[name] = val

        rng = ranges.get(name)
        if rng and len(rng) == 2:
            low, high = float(rng[0]), float(rng[1])
            if not (low <= val <= high):
                warnings.append(
                    "{}={:.4g} 超出训练范围 [{:.4g}, {:.4g}]，属于外推，结果不可信".format(
                        name, val, low, high
                    )
                )

    # 必填检查：一个"分子量"必填项，只要给了它本身、或给了能换算成它的特性粘度，
    # 都算满足。这样用户既可以直接填 Mv_sol，也可以只填 η_sol。
    missing = []
    for name in required:
        if name in cleaned:
            continue
        rule = rules.get(name)
        if rule and rule["eta"] in cleaned:
            continue
        missing.append(name)
    if missing:
        errors.append(
            "缺少必填变量：{}。请把这些变量及其单位、取值范围告知用户并等待其提供，"
            "不要自行编造取值。".format("、".join(missing))
        )

    # 同时提供了分子量和对应的特性粘度：以直接给出的分子量为准（它是模型真正消费的量），
    # 但必须告警，否则用户会以为两个值都参与了计算。
    for target, rule in rules.items():
        eta = rule["eta"]
        if target in cleaned and eta in cleaned:
            warnings.append(
                "同时提供了 {}={:.6g} 与 {}={:.6g}，本次以 {} 为准，{} 未参与换算".format(
                    target, cleaned[target], eta, cleaned[eta], target, eta
                )
            )

    return cleaned, errors, warnings


def build_values(model, provided, meta=None):
    """按拓扑序补齐所有节点，已提供的值优先；特性粘度在此时换算成分子量。

    返回 (values, conversions)：
        values      所有节点/输入的最终取值
        conversions 本次由特性粘度换算出来的分子量，供上层如实告诉用户
                    "这个 Mv 不是他填的，而是由 η 算出来的"

    取值优先级（从高到低）：
        1. 用户直接给出的值（包括 Mv_* 本身）
        2. 由用户给出的 η_* 换算出的分子量
        3. 模型按节点类型推断的值（BN 才有；纯 XGB 没有第 3 档）

    为什么换算必须插在拓扑序里、不能预先统一算完：
        Mv_total 的换算需要 E_total，而 E_total 在 BN 里本身是一个推断节点
        （由 E_insol / E_sol 线性回归得到）。放在拓扑序里，才能保证用到 E_total
        时它已经算好 —— 这也正是原 notebook 里 fill_missing_parents 漏掉的那类依赖。
    """
    values = dict(provided)
    conversions = []
    params, rules = viscosity_setup(meta or {})

    resolved = set()   # 已确定取值的节点，避免重复换算
    visiting = set()   # 环检测：依赖图一旦成环，递归会无限下去

    def ensure(name):
        """确保 name 在 values 里；必要时先换算或按模型推断。"""
        if name in values:
            return values[name]
        if name in resolved:
            return values.get(name)
        if name in visiting:
            raise ValueError("变量 '{}' 的依赖存在环，无法计算".format(name))

        visiting.add(name)
        try:
            # ① 特性粘度换算优先于模型推断：用户给的 η 是实测值，
            #    比 BN 用回归推出来的 Mv 更可信。
            rule = rules.get(name)
            if rule and rule["eta"] in values:
                ethylene = rule["ethylene_input"]
                # W_E 可能本身就是推断节点（E_total），先把它算出来
                if ethylene not in values and ethylene in model:
                    ensure(ethylene)
                if ethylene not in values:
                    raise ValueError(
                        "无法用 {} 换算 {}：缺少乙烯含量 '{}'".format(
                            rule["eta"], name, ethylene))

                mv = eta_to_mv(values[rule["eta"]], values[ethylene], params)
                values[name] = mv
                conversions.append({
                    "eta": rule["eta"],
                    "eta_label": rule.get("eta_label", rule["eta"]),
                    "eta_value": round(float(values[rule["eta"]]), 6),
                    "ethylene_input": ethylene,
                    "ethylene_value": round(float(values[ethylene]), 4),
                    # 标记 W_E 是用户给的还是模型推断的，便于 Agent 说明换算依据
                    "ethylene_source": "user" if ethylene in provided else "model",
                    "target": name,
                    "mw": round(mv, 1),
                })
                resolved.add(name)
                return mv

            # ② 模型节点：先把依赖补齐（父节点、公式输入都要先有值），再按类型算
            if name in model:
                for dep in node_deps(model[name]):
                    if dep in values:
                        continue
                    if dep in model or dep in rules:
                        ensure(dep)
                values[name] = compute_node(model, name, values)
                resolved.add(name)
                return values[name]

            raise ValueError("缺少变量 '{}'".format(name))
        finally:
            visiting.discard(name)

    for node in topo_order(model):
        if node in values:
            continue
        ensure(node)

    return values, conversions


# --------------------------------------------------------------------------
# 命令实现
# --------------------------------------------------------------------------
def cmd_describe(payload):
    """describe 命令：返回节点图与拓扑序，用于排障。

    返回结构与 predict/explain 保持一致（{"ok": True, "data": {...}}），
    否则 client 的 `result.get("ok")` 会把成功结果误判成失败。
    """
    model_id = payload["model_id"]
    model, meta = load_model(model_id)
    roles = {}
    for node in model:
        roles[node] = model[node]["type"]
    graph = {
        node: {
            "type": model[node]["type"],
            "parents": list(model[node].get("parents") or []),
            "deps": node_deps(model[node]),
        }
        for node in model
    }
    return {
        "ok": True,
        "data": {
            "metadata": meta,
            "node_count": len(model),
            "node_roles": roles,
            "graph": graph,
            "topological_order": topo_order(model),
        },
    }


def cmd_predict(payload):
    """predict 命令：校验 → 推断 → 计算全部（或指定的）目标。

    返回的 predictions 与 inferred 是分开的，因为对用户意义不同：
      predictions = 4 个性能指标，是用户真正要的答案
      inferred    = 中间量（Xc/Tm/MI/E_var…），是模型算出来的过程值，
                    回给 LLM 是为了让它能回答"结晶度是多少"这类追问，
                    避免它自己编一个数。
    """
    model_id = payload["model_id"]
    model, meta = load_model(model_id)

    provided, errors, warnings = validate(model, meta, payload.get("values") or {})
    if errors:
        return {"ok": False, "error": "；".join(errors), "warnings": warnings}

    values, conversions = build_values(model, provided, meta)

    target_names = [t["name"] for t in meta.get("targets", [])]
    requested = payload.get("targets") or target_names
    requested = [t for t in requested if t in target_names]
    if not requested:
        return {"ok": False, "error": "没有可预测的目标；可选：{}".format("、".join(target_names))}

    target_meta = {t["name"]: t for t in meta["targets"]}
    predictions = []
    for name in requested:
        item = target_meta[name]
        predictions.append({
            "name": name,
            "label": item.get("label", name),
            "unit": item.get("unit", "-"),
            "value": round(float(values[name]), 4),
            "provided": name in provided,
            "metrics": item.get("metrics", {}),
        })

    inferred = []
    target_set = set(target_names)
    # 由特性粘度换算出来的目标不算"模型推断值"：它的依据是用户的实测 η，
    # 必须与"BN 回归推出来的中间量"区分开，否则 Agent 会说错数据来源。
    converted_targets = {c["target"] for c in conversions}
    for node in topo_order(model):
        if node in target_set or node in provided or node in converted_targets:
            continue
        if model[node]["type"] == "root":
            continue
        label = ""
        unit = "-"
        for item in meta["inputs"]["optional"]:
            if item["name"] == node:
                label, unit = item.get("label", node), item.get("unit", "-")
        inferred.append({
            "name": node,
            "label": label or node,
            "unit": unit,
            "value": round(float(values[node]), 4),
            "node_type": model[node]["type"],
        })

    return {
        "ok": True,
        "data": {
            "model_id": model_id,
            "model_name": meta.get("display_name", model_id),
            "predictions": predictions,
            "inferred": inferred,
            # 本次由特性粘度换算得到的分子量，Agent 要如实告诉用户
            "conversions": conversions,
            "warnings": warnings,
        },
    }


def cmd_explain(payload):
    """explain 命令：对某个目标做 TreeSHAP 解释。

    为什么是"5 个模型各算一次再平均"：
    目标节点有 5 个分折模型，预测值是 5 个模型的平均；如果只解释其中一个，
    解释的和预测的就不是同一套模型。逐个解释再取平均，才能保证
    base_value + Σshap 与 predict 返回的预测值对得上。

    为什么必须先 scaler.transform：
    Booster 是在标准化后的空间上训练的，直接喂原始值会得到完全错误的贡献值。
    但展示给用户的 feature value 用原始值（x_raw），不是标准化值。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    import xgboost as xgb

    model_id = payload["model_id"]
    target = payload.get("target")
    top_k = int(payload.get("top_k") or 8)

    model, meta = load_model(model_id)
    target_names = [t["name"] for t in meta.get("targets", [])]
    if target not in target_names:
        return {"ok": False, "error": "target 必须是 {} 之一".format("、".join(target_names))}

    provided, errors, warnings = validate(model, meta, payload.get("values") or {})
    if errors:
        return {"ok": False, "error": "；".join(errors), "warnings": warnings}

    values, conversions = build_values(model, provided, meta)

    info = model[target]
    if info["type"] != "xgb":
        return {"ok": False, "error": "目标 '{}' 不是 XGBoost 节点，无法用 TreeExplainer".format(target)}

    parents = info["parents"]
    x_raw = np.array([[values[p] for p in parents]], dtype=float)

    fold_values, fold_bases = [], []
    for est, scaler in zip(info["models"], info["scalers"]):
        x_scaled = scaler.transform(x_raw)
        explanation = shap.TreeExplainer(est)(_dmatrix(x_scaled, est))
        fold_values.append(np.ravel(explanation.values))
        fold_bases.append(float(np.ravel(explanation.base_values)[0]))

    shap_values = np.mean(fold_values, axis=0)
    base_value = float(np.mean(fold_bases))
    prediction = float(values[target])
    reconstructed = base_value + float(shap_values.sum())

    label_map = {}
    for item in meta["inputs"]["optional"]:
        label_map[item["name"]] = (item.get("label", item["name"]), item.get("unit", "-"))
    for item in meta["inputs"]["required"]:
        label_map[item["name"]] = (item.get("label", item["name"]), item.get("unit", "-"))

    pairs = []
    for idx, name in enumerate(parents):
        label, unit = label_map.get(name, (name, "-"))
        pairs.append({
            "name": name,
            "label": label,
            "unit": unit,
            "value": round(float(x_raw[0, idx]), 4),
            "shap": round(float(shap_values[idx]), 4),
            "direction": "positive" if shap_values[idx] >= 0 else "negative",
        })
    pairs.sort(key=lambda r: abs(r["shap"]), reverse=True)

    shown = pairs[:top_k]
    other_sum = float(sum(p["shap"] for p in pairs[top_k:]))

    plot_path = None
    try:
        os.makedirs(PLOT_DIR, exist_ok=True)
        plot_path = os.path.join(PLOT_DIR, "{}_{}_{}.png".format(model_id, target, uuid.uuid4().hex[:8]))
        # 图里只用英文节点名，避免中文字体缺失导致方块
        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=x_raw[0],
            feature_names=list(parents),
        )
        shap.plots.waterfall(explanation, max_display=max(top_k, len(parents)), show=False)
        fig = plt.gcf()
        # 宽度给足：分子量/方差量级的数值标签会很长，窄图会挤到画布外
        fig.set_size_inches(11, max(3.5, 0.45 * len(parents) + 1.5))
        fig.tight_layout()
        fig.savefig(plot_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # 画图失败不应影响数值结果
        print("[warn] SHAP 图生成失败：{}".format(exc), file=sys.stderr)
        plot_path = None

    # 由特性粘度换算出来的父节点不算"模型推断"：它的来源是用户实测的 η，
    # 若混进 inferred_parents，Agent 会误说"这个量是模型猜的"。
    converted_targets = {c["target"] for c in conversions}
    inferred_parents = [
        p for p in parents
        if model.get(p, {}).get("type") not in ("root", None)
        and p not in provided
        and p not in converted_targets
    ]

    return {
        "ok": True,
        "data": {
            "model_id": model_id,
            "target": target,
            "target_label": next(
                (t.get("label", target) for t in meta["targets"] if t["name"] == target), target
            ),
            "unit": next((t.get("unit", "-") for t in meta["targets"] if t["name"] == target), "-"),
            "base_value": round(base_value, 4),
            "prediction": round(prediction, 4),
            "reconstructed": round(reconstructed, 4),
            "contributions": shown,
            "other_features_sum": round(other_sum, 4),
            "plot_path": plot_path,
            "inferred_parents": inferred_parents,
            # 若某个父节点是由 η 换算来的，这里会列出，便于 Agent 说清归因对象
            "conversions": conversions,
            "warnings": warnings,
            "note": (
                "SHAP 归因于该目标节点的直接父节点；其中 {} 是由模型推断出来的中间量，"
                "不是用户直接输入的。".format("、".join(inferred_parents))
                if inferred_parents else
                "SHAP 归因于该目标节点的直接父节点。"
            ),
        },
    }


# 命令 → 处理函数。新增一种能力只需在这里注册，client 端加一个同名方法。
# describe/list 是排障用的：agent 侧目前只调 predict/explain。
HANDLERS = {
    "describe": cmd_describe,
    "predict": cmd_predict,
    "explain": cmd_explain,
    "list": lambda payload: {"ok": True, "data": {"models": list_model_ids()}},
}


def main():
    # Windows 下父进程默认按 UTF-8 读，这里强制两端一致，
    # 否则中文变量名/错误信息会变成乱码。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # 协议：stdin 读一个 JSON，stdout 写一个 JSON（且只写一行）。
    raw = sys.stdin.read()
    try:
        payload = json.loads(raw or "{}")
    except Exception as exc:
        print(json.dumps({"ok": False, "error": "请求 JSON 解析失败：{}".format(exc)},
                         ensure_ascii=False))
        return

    handler = HANDLERS.get(payload.get("command"))
    if handler is None:
        print(json.dumps({"ok": False, "error": "未知命令：{}".format(payload.get("command"))},
                         ensure_ascii=False))
        return

    try:
        result = handler(payload)
    except Exception:
        # 兜底：任何未预期异常都转成结构化错误，避免父进程只看到 "exit=1" 而不知道原因。
        # limit=6 只保留最近 6 层栈，够定位又不至于把无关的上层调用刷给用户。
        result = {"ok": False, "error": traceback.format_exc(limit=6)}

    # 单行输出：client 从 stdout 里挑最后一行以 "{" 开头的 JSON 解析。
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
