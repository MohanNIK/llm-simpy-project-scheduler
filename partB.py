# partB.py
# -*- coding: utf-8 -*-
"""
增强版 partB：
- 每个资源等待阶段都会调用 policy.decide_action()（可连接 Qwen API）
- 保持原有产能扰动逻辑、依赖关系、CSV 导出
- 新增：关键线路计算（CPM/最长路）、真·甘特图（关键任务高亮）、导出 critical_path.csv
"""

import simpy
import random
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import os

# ========= 全局配置 =========
NUM_FLOORS = 3
TASKS_PER_FLOOR = ["钢筋", "模板", "机电预留", "浇筑", "养护", "拆模", "砌筑"]
MAX_SIM_DAYS = 200

RANDOM_SEED = 42
ENABLE_DISTURBANCE = True
random.seed(RANDOM_SEED)

BASE_TASK_DUR = {
    "钢筋": 0.2,
    "模板": 0.2,
    "机电预留": 0.1,
    "浇筑": 0.3,
    "养护": 0.0,
    "拆模": 0.2,
    "砌筑": 0.3,
}

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ========= Matplotlib：中文 & 负号 =========
for f in ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC", "Arial Unicode MS"]:
    try:
        matplotlib.rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        pass
matplotlib.rcParams["axes.unicode_minus"] = False

# ========= 基础类 =========
class CapacityServer:
    def __init__(self, env, name, capacity_per_day, delay_sampler, capacity_log, enable_disturb=True):
        self.env = env
        self.name = name
        self.base_capacity = capacity_per_day
        self.capacity_per_day = capacity_per_day
        self.delay_sampler = delay_sampler
        self.capacity_log = capacity_log
        self.enable_disturb = enable_disturb
        self.tokens = simpy.Container(env, init=capacity_per_day, capacity=capacity_per_day)

        env.process(self.refill())
        if self.enable_disturb:
            env.process(self.random_disturbance())

    def _log_capacity(self):
        self.capacity_log.append((self.name, int(self.env.now), int(self.tokens.level)))

    def refill(self):
        self._log_capacity()
        while True:
            yield self.env.timeout(1)
            self._log_capacity()
            need = self.capacity_per_day - self.tokens.level
            if need > 0:
                yield self.tokens.put(need)

    def random_disturbance(self):
        while True:
            yield self.env.timeout(random.randint(10, 20))
            old_cap = self.capacity_per_day
            new_cap = max(1, int(old_cap * random.uniform(0.3, 0.7)))
            lost_tokens = old_cap - new_cap
            if lost_tokens > 0:
                yield self.tokens.get(min(lost_tokens, self.tokens.level))
            yield self.env.timeout(random.randint(2, 4))
            need = old_cap - self.tokens.level
            if need > 0:
                yield self.tokens.put(need)

    def request(self):
        yield self.tokens.get(1)
        delay = self.delay_sampler()
        if delay > 0:
            yield self.env.timeout(delay)
        return delay

# ========= 记录 =========
records = {
    "gantt": [],
    "wip": [],
    "delay_breakdown": defaultdict(lambda: defaultdict(float))
}
in_progress = set()
starts = []
finishes = []
capacity_log = []

# ========= 依赖图构造（用于关键线路） =========
def _build_dependencies():
    """
    返回 deps: {task_label: [pred_task_labels, ...]}
    规则：
      - 同层顺序依赖（钢筋->模板->机电预留->浇筑->养护->拆模->砌筑）
      - 跨层依赖：F(n-1)-模板 -> F(n)-浇筑
    """
    deps = defaultdict(list)
    for floor in range(1, NUM_FLOORS + 1):
        for i, tn in enumerate(TASKS_PER_FLOOR):
            cur = f"F{floor}-{tn}"
            if i > 0:
                prev = f"F{floor}-{TASKS_PER_FLOOR[i-1]}"
                deps[cur].append(prev)
        if floor > 1:
            deps[f"F{floor}-浇筑"].append(f"F{floor-1}-模板")
    return deps

# ========= 关键线路计算（最长路/CPM） =========
def compute_critical_path(gantt_df: pd.DataFrame):
    """
    基于 Start/Finish 和依赖图求关键线路：
    返回 (critical_set, path_list, total_duration)
    """
    if gantt_df.empty:
        return set(), [], 0.0

    # 用仿真得到的真实工期
    dur = {row["Task"]: float(row["Finish"] - row["Start"]) for _, row in gantt_df.iterrows()}
    deps = _build_dependencies()

    nodes = set(dur.keys())
    succs = defaultdict(list)
    indeg = defaultdict(int)
    for v in nodes:
        for u in deps.get(v, []):
            succs[u].append(v)
        indeg[v] = len(deps.get(v, []))

    # 拓扑 DP（最长路径）
    dist = {v: -1e18 for v in nodes}
    parent = {v: None for v in nodes}
    from collections import deque as _dq
    q0 = _dq([v for v in nodes if indeg[v] == 0])
    for v in list(q0):
        dist[v] = dur.get(v, 0.0)

    while q0:
        u = q0.popleft()
        for v in succs.get(u, []):
            cand = dist[u] + dur.get(v, 0.0)
            if cand > dist[v]:
                dist[v] = cand
                parent[v] = u
            indeg[v] -= 1
            if indeg[v] == 0:
                q0.append(v)

    end = max(nodes, key=lambda x: dist.get(x, -1e18))
    total = dist.get(end, 0.0)

    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return set(path), path, total

# ========= WIP 监控 =========
def wip_monitor(env):
    while True:
        records["wip"].append((env.now, len(in_progress)))
        yield env.timeout(1)

# ========= 主任务流程 =========
def task_process(env, floor, task_name, crane, rfi, change, rework, policy=None, dependencies=None):
    """
    关键修正：真正的开始时间应在等待所有依赖完成之后再记录。
    """
    if dependencies is None:
        dependencies = []
    label = f"F{floor}-{task_name}"

    delay_detail = {"RFI": 0.0, "变更": 0.0, "塔吊": 0.0, "返工": 0.0}

    # 1) 先等依赖完成（这里任务尚未开始）
    for dep in dependencies:
        if dep is not None:
            yield dep

    # 2) 依赖全部满足，此刻才是真正“开始”
    start_time = env.now
    in_progress.add(label)
    starts.append((label, start_time))

    # ==== RFI ====
    if policy and hasattr(policy, "decide_action"):
        ctx = {"current_task": label, "time": env.now, "resource": "RFI", "in_progress": list(in_progress)}
        decision = policy.decide_action(ctx)
        print(f"[LLMPolicy] 对 {label} 的 RFI 阶段决策: {decision}")
    rfi_start = env.now
    yield env.process(rfi.request())
    delay_detail["RFI"] += env.now - rfi_start

    # ==== 变更 ====
    if policy and hasattr(policy, "decide_action"):
        ctx = {"current_task": label, "time": env.now, "resource": "变更", "in_progress": list(in_progress)}
        decision = policy.decide_action(ctx)
        print(f"[LLMPolicy] 对 {label} 的 变更 阶段决策: {decision}")
    change_start = env.now
    if random.random() < 0.25:
        yield env.process(change.request())
    delay_detail["变更"] += env.now - change_start

    # ==== 塔吊 ====
    if policy and hasattr(policy, "decide_action"):
        ctx = {"current_task": label, "time": env.now, "resource": "塔吊", "in_progress": list(in_progress)}
        decision = policy.decide_action(ctx)
        print(f"[LLMPolicy] 对 {label} 的 塔吊 阶段决策: {decision}")
    crane_start = env.now
    yield env.process(crane.request())
    delay_detail["塔吊"] += env.now - crane_start

    # ==== 返工 ====
    if policy and hasattr(policy, "decide_action"):
        ctx = {"current_task": label, "time": env.now, "resource": "返工", "in_progress": list(in_progress)}
        decision = policy.decide_action(ctx)
        print(f"[LLMPolicy] 对 {label} 的 返工 阶段决策: {decision}")
    if random.random() < 0.15:
        rework_start = env.now
        yield env.process(rework.request())
        delay_detail["返工"] += env.now - rework_start

    # ==== 基础工期 ====
    base_dur = BASE_TASK_DUR.get(task_name, 0.0)
    if base_dur > 0:
        yield env.timeout(base_dur)

    finish_time = env.now
    records["gantt"].append((label, start_time, finish_time))
    finishes.append((label, finish_time))
    in_progress.discard(label)

    for k, v in delay_detail.items():
        records["delay_breakdown"][(floor, task_name)][k] += v

# ========= 仿真主函数 =========
def run_simulation(policy=None):
    env = simpy.Environment()

    crane = CapacityServer(env, "塔吊", 1, lambda: random.uniform(0.5, 1.5), capacity_log, ENABLE_DISTURBANCE)
    rfi   = CapacityServer(env, "RFI",  4, lambda: random.uniform(2, 4), capacity_log, ENABLE_DISTURBANCE)
    change= CapacityServer(env, "变更", 1, lambda: random.uniform(4, 7), capacity_log, ENABLE_DISTURBANCE)
    rework= CapacityServer(env, "返工", 2, lambda: random.uniform(1, 2), capacity_log, ENABLE_DISTURBANCE)

    env.process(wip_monitor(env))

    task_refs_prev_floor = {}
    for floor in range(1, NUM_FLOORS + 1):
        task_refs = {}
        for i, task_name in enumerate(TASKS_PER_FLOOR):
            deps = []
            if i > 0:
                deps.append(task_refs[TASKS_PER_FLOOR[i - 1]])
            if task_name == "浇筑" and floor > 1:
                deps.append(task_refs_prev_floor.get("模板"))
            proc = env.process(task_process(env, floor, task_name, crane, rfi, change, rework, policy, deps))
            task_refs[task_name] = proc
        task_refs_prev_floor = task_refs

    env.run(until=MAX_SIM_DAYS)

def run_simulation_with_conflicts(policy=None):
    return run_simulation(policy=policy)

# ========= 图表 =========
def _plot_wip(wip_df: pd.DataFrame, out_path: str):
    if wip_df.empty:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(wip_df["Day"], wip_df["WIP"])
    plt.title("WIP 随时间变化")
    plt.xlabel("Day")
    plt.ylabel("WIP")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

def _plot_gantt(gantt_df: pd.DataFrame, out_path: str):
    """
    真·甘特：按时间轴绘制；关键任务红色，非关键灰色
    """
    if gantt_df.empty:
        return

    # 关键线路
    critical_set, path, total = compute_critical_path(gantt_df)

    df = gantt_df.copy()
    df["Duration"] = df["Finish"] - df["Start"]
    df = df.sort_values(["Start", "Finish"]).reset_index(drop=True)

    plt.figure(figsize=(12, max(4, 0.35 * len(df))))
    h = 0.8
    y_ticks, y_pos = [], []

    for i, row in df.iterrows():
        task = row["Task"]
        left = row["Start"]
        width = row["Duration"]
        y = i
        color = "#d62728" if task in critical_set else "#7f7f7f"
        plt.broken_barh([(left, width)], (y - h/2, h),
                        facecolors=color, edgecolor="k", linewidth=0.3)
        y_ticks.append(task)
        y_pos.append(y)

    plt.yticks(y_pos, y_ticks)
    plt.xlabel("Day")
    plt.title("Gantt Chart（含关键线路高亮）")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor="#d62728", edgecolor="k", label="关键任务"),
               Patch(facecolor="#7f7f7f", edgecolor="k", label="非关键任务")]
    plt.legend(handles=handles, loc="upper right")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

# ========= 输出 =========
def save_results():
    gantt_df = pd.DataFrame(records["gantt"], columns=["Task", "Start", "Finish"])
    wip_df = pd.DataFrame(records["wip"], columns=["Day", "WIP"])
    delay_df = pd.DataFrame([
        {"Task": f"{floor}-{task}", **delays}
        for (floor, task), delays in records["delay_breakdown"].items()
    ]).fillna(0)
    cap_df = pd.DataFrame(capacity_log, columns=["Resource", "Day", "Capacity"]).drop_duplicates()

    # 关键线路计算与导出
    critical_set, critical_path, total = compute_critical_path(gantt_df)
    crit_df = pd.DataFrame({
        "Order": list(range(1, len(critical_path)+1)),
        "Task": critical_path
    })
    crit_df["TotalDuration"] = total if len(crit_df) > 0 else 0.0

    gantt_csv   = os.path.join(OUT_DIR, "results_gantt.csv")
    wip_csv     = os.path.join(OUT_DIR, "results_wip.csv")
    delay_csv   = os.path.join(OUT_DIR, "delay_breakdown.csv")
    starts_csv  = os.path.join(OUT_DIR, "starts.csv")
    finishes_csv= os.path.join(OUT_DIR, "finishes.csv")
    caps_csv    = os.path.join(OUT_DIR, "capacities.csv")
    crit_csv    = os.path.join(OUT_DIR, "critical_path.csv")

    gantt_df.to_csv(gantt_csv, index=False)
    wip_df.to_csv(wip_csv, index=False)
    delay_df.to_csv(delay_csv, index=False)
    pd.DataFrame(starts, columns=["Task", "Start"]).to_csv(starts_csv, index=False)
    pd.DataFrame(finishes, columns=["Task", "Finish"]).to_csv(finishes_csv, index=False)
    cap_df.to_csv(caps_csv, index=False)
    crit_df.to_csv(crit_csv, index=False)

    _plot_wip(wip_df, os.path.join(OUT_DIR, "wip.png"))
    _plot_gantt(gantt_df, os.path.join(OUT_DIR, "gantt.png"))

    print(f"=== B 模型运行完成 ===")
    print(f"任务总数: {len(gantt_df)}")
    if not gantt_df.empty:
        print(f"最后完成时间: {gantt_df['Finish'].max():.2f} 天")
    print(f"- 甘特 CSV: {gantt_csv}")
    print(f"- WIP CSV: {wip_csv}")
    print(f"- 延误分解 CSV: {delay_csv}")
    print(f"- starts/finishes: {starts_csv}, {finishes_csv}")
    print(f"- 资源容量 CSV: {caps_csv}")
    print(f"- 关键线路 CSV: {crit_csv}")
    print(f"- 图表: {os.path.join(OUT_DIR, 'gantt.png')}, {os.path.join(OUT_DIR, 'wip.png')}")

if __name__ == "__main__":
    run_simulation()
    save_results()
