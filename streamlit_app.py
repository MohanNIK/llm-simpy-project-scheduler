# streamlit_app.py
# 运行：  streamlit run streamlit_app.py
import sys
print(sys.executable)
import os
import importlib
import types
import random
import simpy
import streamlit as st
import pandas as pd
import inspect

# ---- 基本路径 ----
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(CUR_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

st.set_page_config(page_title="施工调度仿真（LLM+SimPy）", layout="wide")
st.title("施工调度仿真（LLM + SimPy）")

# ====== 左侧栏：参数输入 ======
with st.sidebar:
    st.header("仿真参数")
    num_floors = st.number_input("楼层数", min_value=1, max_value=50, value=3, step=1)
    max_days   = st.number_input("最大仿真天数", min_value=10, max_value=3650, value=200, step=10)
    seed       = st.number_input("随机种子", min_value=0, max_value=100000, value=42, step=1)
    enable_disturb = st.checkbox("启用随机扰动（天气/审批缺席等）", value=True)

    st.subheader("资源日产能（token/天）")
    cap_crane  = st.number_input("塔吊", min_value=1, max_value=20, value=1, step=1)
    cap_rfi    = st.number_input("RFI", min_value=1, max_value=50, value=4, step=1)
    cap_change = st.number_input("变更", min_value=1, max_value=50, value=1, step=1)
    cap_rework = st.number_input("返工", min_value=1, max_value=50, value=2, step=1)

    st.subheader("策略选择")
    strategy_name = st.selectbox("选择策略", ["无（不启用策略）", "LLMPolicy"], index=1)

    st.subheader("LLM 设置（Qwen）")
    qwen_key = st.text_input("DASHSCOPE_API_KEY / Qwen API Key（仅本机填写）", type="password")
    proxy_host = st.text_input("HTTP 代理主机（可留空）", value="")
    proxy_port = st.text_input("HTTP 代理端口（可留空）", value="")

    run_btn = st.button("运行仿真", use_container_width=True)

# ====== 工具函数 ======
def normalize_policy(p):
    """把 list/tuple 等转成单个策略对象；不合格就返回 None。"""
    if p is None:
        return None
    if isinstance(p, (list, tuple, set)):
        for x in p:
            if hasattr(x, "decide_action") and callable(getattr(x, "decide_action")):
                return x
        return None
    return p if hasattr(p, "decide_action") and callable(getattr(p, "decide_action")) else None

def inject_env_from_ui():
    """在导入 LLMPolicy/partB 之前注入代理和 Key（关键步骤）"""
    # 代理
    if proxy_host.strip() and proxy_port.strip():
        os.environ["HTTP_PROXY"]  = f"http://{proxy_host.strip()}:{proxy_port.strip()}"
        os.environ["HTTPS_PROXY"] = f"http://{proxy_host.strip()}:{proxy_port.strip()}"
        st.info(f"[代理] 已启用 HTTP 代理 {proxy_host}:{proxy_port}")
    else:
        # 清理可能的历史残留
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ.pop(k, None)

    # Qwen Key
    if qwen_key.strip():
        os.environ["DASHSCOPE_API_KEY"] = qwen_key.strip()
        os.environ["QWEN_API_KEY"]      = qwen_key.strip()  # 兼容
        st.info("已注入 Qwen API Key（仅当前进程）")
    else:
        os.environ.pop("DASHSCOPE_API_KEY", None)
        os.environ.pop("QWEN_API_KEY", None)
        st.warning("未设置 Qwen API Key：若 LLMPolicy 未硬编码 Key，将无法外网调用。")

def load_modules_fresh():
    """删除模块缓存后再导入，确保拿到最新环境变量"""
    for m in ["policies.llm_policy", "partB"]:
        if m in sys.modules:
            del sys.modules[m]

    partB = importlib.import_module("partB")

    LLMPolicy = None
    if strategy_name == "LLMPolicy":
        try:
            llm_mod = importlib.import_module("policies.llm_policy")
            LLMPolicy = getattr(llm_mod, "LLMPolicy")
        except Exception as e:
            st.error(f"无法加载 LLMPolicy：{e}")
            LLMPolicy = None
    return partB, LLMPolicy

def apply_overrides(partB):
    # 覆盖 partB 的全局参数
    partB.NUM_FLOORS = int(num_floors)
    partB.MAX_SIM_DAYS = int(max_days)
    partB.RANDOM_SEED = int(seed)
    partB.ENABLE_DISTURBANCE = bool(enable_disturb)

    # 通过全局变量把 GUI 的产能配置传给 run_simulation 的 patch
    partB._GUI_CAPS = {
        "crane":  cap_crane,
        "rfi":    cap_rfi,
        "change": cap_change,
        "rework": cap_rework,
    }

def patch_partB_run(partB):
    """在 partB.run_simulation 中读取 _GUI_CAPS，并将 policy 智能透传给 task_process"""
    if hasattr(partB, "_RUN_PATCHED"):
        return  # 已经打过补丁

    orig_run = partB.run_simulation

    def run_simulation_patched(*args, **kwargs):
        caps = getattr(partB, "_GUI_CAPS", None)
        if caps is None:
            return orig_run(*args, **kwargs)

        from partB import (
            CapacityServer, wip_monitor, task_process, TASKS_PER_FLOOR,
            NUM_FLOORS, MAX_SIM_DAYS, ENABLE_DISTURBANCE, capacity_log
        )

        env = simpy.Environment()
        random.seed(partB.RANDOM_SEED)
        env.process(wip_monitor(env))

        # 资源（使用 GUI 的配置）
        crane  = CapacityServer(env, "塔吊", caps["crane"],  lambda: random.uniform(0.5, 1.5), capacity_log, ENABLE_DISTURBANCE)
        rfi    = CapacityServer(env, "RFI",  caps["rfi"],   lambda: random.uniform(2, 4),      capacity_log, ENABLE_DISTURBANCE)
        change = CapacityServer(env, "变更", caps["change"], lambda: random.uniform(4, 7),      capacity_log, ENABLE_DISTURBANCE)
        rework = CapacityServer(env, "返工", caps["rework"], lambda: random.uniform(1, 2),      capacity_log, ENABLE_DISTURBANCE)

        # 策略对象（可能为 None）
        policy = normalize_policy(kwargs.get("policy", None))

        # 反射 task_process 的签名，决定是否传 policy / dependencies
        sig = inspect.signature(task_process)
        has_policy = "policy" in sig.parameters
        has_dependencies = "dependencies" in sig.parameters

        task_refs_prev_floor = {}
        for floor in range(1, NUM_FLOORS + 1):
            task_refs = {}
            for i, task_name in enumerate(TASKS_PER_FLOOR):
                deps = []
                if i > 0:
                    deps.append(task_refs[TASKS_PER_FLOOR[i - 1]])
                if task_name == "浇筑" and floor > 1:
                    deps.append(task_refs_prev_floor.get("模板"))

                # 组装调用参数
                call_kwargs = {}
                if has_policy:
                    call_kwargs["policy"] = policy
                if has_dependencies:
                    call_kwargs["dependencies"] = deps

                # 兼容旧签名（没有 policy/或者 dependencies 不是关键字参数）
                try:
                    if has_policy and has_dependencies:
                        proc = env.process(task_process(env, floor, task_name, crane, rfi, change, rework, **call_kwargs))
                    elif has_policy and not has_dependencies:
                        # 旧代码可能把 deps 当作位置参数
                        proc = env.process(task_process(env, floor, task_name, crane, rfi, change, rework, deps, **{"policy": policy}))
                    elif not has_policy and has_dependencies:
                        proc = env.process(task_process(env, floor, task_name, crane, rfi, change, rework, **{"dependencies": deps}))
                    else:
                        # 最旧签名：task_process(env, floor, task_name, crane, rfi, change, rework, deps)
                        proc = env.process(task_process(env, floor, task_name, crane, rfi, change, rework, deps))
                except TypeError:
                    # 再次兜底：尝试最朴素的调用（避免因签名差异崩溃）
                    try:
                        proc = env.process(task_process(env, floor, task_name, crane, rfi, change, rework, deps))
                    except Exception as e:
                        raise e

                task_refs[task_name] = proc
            task_refs_prev_floor = task_refs

        env.run(until=MAX_SIM_DAYS)

    partB.run_simulation = types.FunctionType(
        run_simulation_patched.__code__, globals(), "run_simulation_patched",
        closure=run_simulation_patched.__closure__
    )
    partB._RUN_PATCHED = True

def inject_proxy_and_key():
    """向后兼容：保留你的原函数名，内部调用新逻辑"""
    inject_env_from_ui()

# ========= 运行 =========
if run_btn:
    # 1) 先注入环境变量（代理 + Key）
    inject_env_from_ui()

    # 2) 再“干净地”导入 partB 与 LLMPolicy（确保拿到新环境）
    partB, LLMPolicy = load_modules_fresh()

    # 3) 覆盖参数 & 可选补丁
    apply_overrides(partB)
    patch_partB_run(partB)  # 如果你的 partB 本就支持从 _GUI_CAPS 读取，可移除此行

    # 4) 初始化策略（如启用）
    policy = None
    if strategy_name == "LLMPolicy" and LLMPolicy is not None:
        try:
            policy = LLMPolicy()   # 会在 __init__ 中读取环境变量并创建 httpx/OpenAI 客户端
            if hasattr(policy, "before_simulation"):
                policy.before_simulation()
        except Exception as e:
            st.error(f"初始化 LLMPolicy 失败：{e}")
            policy = None

    # 5) 执行仿真
    try:
        if policy:
            partB.run_simulation(policy=policy)
        else:
            partB.run_simulation()
        partB.save_results()
    except Exception as e:
        st.exception(e)

    # 6) 展示结果
    st.success("仿真完成 ✅")
    col1, col2 = st.columns(2)
    with col1:
        png_gantt = os.path.join(RESULTS_DIR, "gantt.png")
        if os.path.exists(png_gantt):
            st.image(png_gantt, caption="Gantt Chart")
        csv_gantt = os.path.join(RESULTS_DIR, "results_gantt.csv")
        if os.path.exists(csv_gantt):
            st.dataframe(pd.read_csv(csv_gantt).head(50))

    with col2:
        png_wip = os.path.join(RESULTS_DIR, "wip.png")
        if os.path.exists(png_wip):
            st.image(png_wip, caption="WIP Over Time")
        csv_delay = os.path.join(RESULTS_DIR, "delay_breakdown.csv")
        if os.path.exists(csv_delay):
            st.dataframe(pd.read_csv(csv_delay).head(50))
