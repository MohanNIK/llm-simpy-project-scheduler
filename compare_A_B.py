# compare_A_B.py
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import os

# ==== 参数 ====
N_RUNS = 5  # 每个模型重复次数
RESULTS_DIR = "./compare_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_model(label, script_path):
    makespans = []
    delay_breakdowns = []
    wip_runs = []

    for i in range(N_RUNS):
        print(f"[{label}] 运行第 {i+1}/{N_RUNS} 次...")
        subprocess.run(["python", script_path], check=True)

        try:
            gantt = pd.read_csv("results_gantt.csv")
            wip = pd.read_csv("results_wip.csv")
            delay_df = pd.read_csv("delay_breakdown.csv")

            makespan = gantt["Finish"].max()
            makespans.append(makespan)
            delay_breakdowns.append(delay_df.drop(columns=["Task"]).sum())
            wip_runs.append(wip)

            # 保存每次的原始结果
            gantt.to_csv(f"{RESULTS_DIR}/{label}_gantt_run{i+1}.csv", index=False)
            wip.to_csv(f"{RESULTS_DIR}/{label}_wip_run{i+1}.csv", index=False)
            delay_df.to_csv(f"{RESULTS_DIR}/{label}_delay_run{i+1}.csv", index=False)

        except Exception as e:
            print(f"读取结果失败：{e}")

    avg_makespan = sum(makespans) / len(makespans)
    avg_delay_breakdown = pd.concat(delay_breakdowns, axis=1).mean(axis=1)
    avg_wip = pd.concat(wip_runs).groupby("Day")["WIP"].mean()

    return avg_makespan, avg_delay_breakdown, avg_wip

# ==== 主流程 ====
print("=== 运行 A 模型 ===")
a_makespan, a_delay, a_wip = run_model("A", "partA.py")

print("=== 运行 B 模型 ===")
b_makespan, b_delay, b_wip = run_model("B", "partB.py")

# ==== 工期对比 ====
plt.figure()
plt.bar(["A 模型", "B 模型"], [a_makespan, b_makespan], color=["skyblue", "salmon"])
plt.ylabel("平均工期 (天)")
plt.title("A vs B 平均工期对比")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/makespan_compare.png", dpi=180)
plt.close()

# ==== 延误占比对比 ====
delay_df = pd.DataFrame({"A": a_delay, "B": b_delay})
delay_df.plot(kind="bar")
plt.ylabel("平均延误天数")
plt.title("A vs B 延误来源对比")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/delay_breakdown_compare.png", dpi=180)
plt.close()

# ==== WIP 曲线对比 ====
plt.figure()
plt.plot(a_wip.index, a_wip.values, label="A 模型", marker="o")
plt.plot(b_wip.index, b_wip.values, label="B 模型", marker="s")
plt.xlabel("Day")
plt.ylabel("平均 WIP")
plt.title("A vs B 平均 WIP 曲线")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/wip_compare.png", dpi=180)
plt.close()

print("✅ 对比实验完成，结果已保存到 compare_results 目录。")
