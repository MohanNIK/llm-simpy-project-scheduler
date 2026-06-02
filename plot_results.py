# plot_results.py - 一键版，支持延误分解绘图
import os, glob
import pandas as pd
import matplotlib.pyplot as plt

RES_DIR = "./results"
FIG_DIR = os.path.join(RES_DIR, "figs")
os.makedirs(FIG_DIR, exist_ok=True)

def try_get(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"这些列名都没找到：{candidates}；请检查 CSV 列名并在脚本里改。")
    return None

# 1) WIP 曲线
wip_path = os.path.join(RES_DIR, "wip.csv")
if os.path.exists(wip_path):
    wip = pd.read_csv(wip_path)
    tcol = try_get(wip, ["day", "time", "t"])
    wcol = try_get(wip, ["wip", "WIP", "in_system"])
    plt.figure()
    plt.plot(wip[tcol], wip[wcol], linewidth=1.8)
    plt.xlabel(tcol)
    plt.ylabel("WIP")
    plt.title("WIP Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "wip_over_time.png"), dpi=180)
    plt.close()

# 2) 累计完工 S 曲线
fin_path = os.path.join(RES_DIR, "finish_times.csv")
if os.path.exists(fin_path):
    ft = pd.read_csv(fin_path)
    fcol = try_get(ft, ["finish_day", "finish_time", "finish", "end_day"])
    ft_sorted = ft.sort_values(fcol)
    ft_sorted["cum_done"] = range(1, len(ft_sorted)+1)
    plt.figure()
    plt.step(ft_sorted[fcol], ft_sorted["cum_done"], where="post")
    plt.xlabel("Time (day)")
    plt.ylabel("Cumulative finished units")
    plt.title("S-curve (Cumulative Completions)")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "s_curve.png"), dpi=180)
    plt.close()

    # 2b) 单元历时分布
    dcol = ft.columns.intersection(["duration_day","duration","flow_time","lt"]).tolist()
    if not dcol:
        scol = try_get(ft, ["start_day","start_time","start"], required=False)
        if scol:
            ft["__dur__"] = ft[fcol] - ft[scol]
            dname = "__dur__"
        else:
            dname = None
    else:
        dname = dcol[0]

    if dname is not None:
        plt.figure()
        ft[dname].plot(kind="hist", bins=20, alpha=0.8)
        plt.xlabel("Unit duration (days)")
        plt.ylabel("Count")
        plt.title("Distribution of Unit Flow Time")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "unit_duration_hist.png"), dpi=180)
        plt.close()

# 3) 排队：找 top3 平均队长
queue_files = glob.glob(os.path.join(RES_DIR, "queue_*.csv"))
if queue_files:
    stats = []
    traces = []
    for qp in queue_files:
        q = pd.read_csv(qp)
        tcol = q.columns.intersection(["day","time","t"]).tolist()
        qcol = q.columns.intersection(["queue_len","q_len","len","queue"]).tolist()
        if not tcol or not qcol: 
            continue
        tcol, qcol = tcol[0], qcol[0]
        mean_q = q[qcol].mean()
        stats.append((os.path.basename(qp), mean_q))
        traces.append((os.path.basename(qp), q[[tcol,qcol]].rename(columns={tcol:"t",qcol:"q"})))
    if stats:
        stats.sort(key=lambda x: x[1], reverse=True)
        top3 = [s[0] for s in stats[:3]]
        plt.figure()
        for name, df in traces:
            if name in top3:
                df = df.sort_values("t")
                plt.plot(df["t"], df["q"], label=name.replace(".csv",""))
        plt.xlabel("Time (day)")
        plt.ylabel("Queue length")
        plt.title("Top-3 Queues by Average Backlog")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "queues_top3.png"), dpi=180)
        plt.close()

# 4) 延误分解统计（新增）
delay_path = os.path.join(RES_DIR, "delay_breakdown.csv")
if os.path.exists(delay_path):
    db = pd.read_csv(delay_path)

    # 自动匹配列名（原因列 & 天数列）
    reason_col = try_get(db, ["reason", "原因", "delay_reason", "type", "类别"])
    days_col = try_get(db, ["days", "延误天数", "total_days", "delay_days", "duration", "总天数"])

    # 饼图
    plt.figure()
    plt.pie(db[days_col], labels=db[reason_col], autopct='%1.1f%%', startangle=140)
    plt.title("Delay Breakdown by Reason")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "delay_breakdown_pie.png"), dpi=180)
    plt.close()

    # 柱状图
    plt.figure()
    db_sorted = db.sort_values(days_col, ascending=False)
    plt.bar(db_sorted[reason_col], db_sorted[days_col])
    plt.ylabel("Delay Days")
    plt.title("Delay Breakdown (Total Days by Reason)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "delay_breakdown_bar.png"), dpi=180)
    plt.close()

