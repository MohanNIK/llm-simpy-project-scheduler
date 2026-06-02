# partC-llm.py
# -*- coding: utf-8 -*-

import os
import sys
import importlib
import socket
import traceback
import httpx

DASHSCOPE_HOST = "dashscope.aliyuncs.com"
DASHSCOPE_PORT = 443

# ==== 自动注入 Clash HTTP 代理（默认 127.0.0.1:7890）====
def enable_clash_http_proxy(default_port=7890):
    host = "127.0.0.1"
    port = default_port
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((host, port)) == 0:
                os.environ["HTTP_PROXY"]  = f"http://{host}:{port}"
                os.environ["HTTPS_PROXY"] = f"http://{host}:{port}"
                print(f"[代理] 已启用 HTTP 代理 {host}:{port}")
            else:
                print(f"[代理] 警告：Clash HTTP 代理端口 {port} 未开放，跳过注入。")
    except Exception as e:
        print(f"[代理] 代理注入失败：{e}")

def probe_dashscope_connectivity(timeout=5.0):
    # 1) TCP 级连通性
    try:
        with socket.create_connection((DASHSCOPE_HOST, DASHSCOPE_PORT), timeout=timeout):
            print("[连通性] TCP 直连 DashScope: ✅")
    except Exception as e:
        print(f"[连通性] TCP 直连失败（可能正常，因为需走代理）：{e}")

    # 2) HTTP 层（走当前环境代理）
    try:
        proxies = {
            "http://": os.getenv("HTTP_PROXY"),
            "https://": os.getenv("HTTPS_PROXY"),
        }
        proxies = {k: v for k, v in proxies.items() if v}
        with httpx.Client(proxies=proxies or None, timeout=timeout) as client:
            r = client.get("https://dashscope.aliyuncs.com/healthz")
            print(f"[连通性] HTTPS GET /healthz -> {r.status_code}")
    except Exception as e:
        print(f"[连通性] HTTPS 探测失败：{e}")

enable_clash_http_proxy()
probe_dashscope_connectivity()

# ==== 路径设置 ====
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
POLICIES_DIR = os.path.join(CUR_DIR, "policies")
for p in (CUR_DIR, POLICIES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# ==== 导入仿真 ====
try:
    from partB import run_simulation, save_results
except ImportError as e:
    raise ImportError(f"导入 partB 失败：{e}")

# ==== 导入 LLM 策略 ====
try:
    llm_module = importlib.import_module("policies.llm_policy")
    LLMPolicy = getattr(llm_module, "LLMPolicy")
except Exception as e:
    raise ImportError(f"无法加载 policies.llm_policy 中的 LLMPolicy：{e}")

def _check_api_key():
    api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        print("[警告] 未在环境变量中发现 Qwen API Key（DASHSCOPE_API_KEY 或 QWEN_API_KEY）。"
              "如果 llm_policy.py 内没有硬编码 key，将无法成功调用 Qwen。")
    return api_key

def main():
    _check_api_key()

    # 实例化策略
    try:
        policy = LLMPolicy()
    except Exception:
        print("[错误] 实例化 LLMPolicy 失败：")
        traceback.print_exc()
        raise

    # 校验接口
    if not hasattr(policy, "decide_action") or not callable(getattr(policy, "decide_action")):
        raise AttributeError("LLMPolicy 未实现 decide_action(context: dict) 方法。")

    # 仿真前
    if hasattr(policy, "before_simulation"):
        policy.before_simulation()

    print("[INFO] 即将启动仿真，并强制传入 LLMPolicy（所有阶段都会尝试调用 decide_action）。")

    # 运行仿真
    try:
        run_simulation(policy=policy)
        save_results()
    except Exception:
        print("[错误] 仿真过程中出现异常：")
        traceback.print_exc()
        raise

    results_paths = {
        "gantt": os.path.join(CUR_DIR, "results", "results_gantt.csv"),
        "wip": os.path.join(CUR_DIR, "results", "results_wip.csv"),
        "delay": os.path.join(CUR_DIR, "results", "delay_breakdown.csv"),
    }

    # 仿真后
    if hasattr(policy, "after_simulation"):
        try:
            policy.after_simulation(results_paths)
        except Exception:
            print("[警告] after_simulation 执行报错（不影响结果文件生成）：")
            traceback.print_exc()

    print("\n=== partC-llm.py 运行完成 ===")
    for k, v in results_paths.items():
        exists = "✓" if os.path.exists(v) else "✗"
        print(f"{k:6s}: {v} {exists}")

if __name__ == "__main__":
    main()
