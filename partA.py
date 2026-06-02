# partA_delay_breakdown.py
import os, math, random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import simpy

# ========= 工具函数 =========
RNG = np.random.default_rng(42)
random.seed(42)

def u_int(a: int, b: int) -> int:
    return int(RNG.integers(a, b + 1))

def u_float(a: float, b: float) -> float:
    return float(RNG.uniform(a, b))

def exp_days(mean_days: float) -> float:
    if mean_days <= 0:
        return 0.0
    return float(RNG.exponential(mean_days))

def bernoulli(p: float) -> bool:
    return RNG.random() < p

# ========= 扰动管理 =========
@dataclass
class Shock:
    day: int
    link_id: str
    capacity_multiplier: float = 1.0
    extra_delay_days: float = 0.0

class ShockManager:
    def __init__(self, shocks: List[Shock]):
        self.by_day: Dict[int, List[Shock]] = {}
        for s in shocks:
            self.by_day.setdefault(s.day, []).append(s)

    def today_adjustment(self, day: int, link_id: str) -> Tuple[float, float]:
        mult, extra = 1.0, 0.0
        for s in self.by_day.get(day, []):
            if s.link_id == link_id:
                mult *= s.capacity_multiplier
                extra += s.extra_delay_days
        return mult, extra

# ========= 容量服务器 =========
class CapacityServer:
    def __init__(self, env, link_id, base_capacity_per_day, delay_sampler, p_fail=0.0, shock_mgr=None):
        self.env = env
        self.link_id = link_id
        self.base_capacity_per_day = base_capacity_per_day
        self.delay_sampler = delay_sampler
        self.p_fail = p_fail
        self.shock_mgr = shock_mgr
        self.bucket = simpy.Container(env, init=0, capacity=1e9)
        self.queue_len = 0
        self.queue_time_series = []
        env.process(self._daily_refill())

    def _daily_refill(self):
        day = 0
        while True:
            mult, _ = (1.0, 0.0)
            if self.shock_mgr:
                mult, _ = self.shock_mgr.today_adjustment(day, self.link_id)
            quota = max(0.0, self.base_capacity_per_day * mult)
            yield self.bucket.put(quota)
            day += 1
            yield self.env.timeout(1.0)

    def request_one(self):
        self.queue_len += 1
        self.queue_time_series.append((math.floor(self.env.now), self.queue_len))
        yield self.bucket.get(1.0)
        self.queue_len -= 1
        self.queue_time_series.append((math.floor(self.env.now), self.queue_len))

        delay = self.delay_sampler()
        if self.shock_mgr:
            day = math.floor(self.env.now)
            _, extra_delay = self.shock_mgr.today_adjustment(day, self.link_id)
            delay += extra_delay
        if delay > 0:
            yield self.env.timeout(delay)

    def fail(self):
        return bernoulli(self.p_fail)

# ========= 物资供应 =========
class MaterialPipeline:
    def __init__(self, env, link_id, base_amount_per_day, shock_mgr=None, p_fail=0.0):
        self.env = env
        self.link_id = link_id
        self.base_amount_per_day = base_amount_per_day
        self.shock_mgr = shock_mgr
        self.p_fail = p_fail
        self.pool = simpy.Container(env, init=0.0, capacity=1e9)
        env.process(self._daily_supply())

    def _daily_supply(self):
        day = 0
        while True:
            if self.p_fail and bernoulli(self.p_fail):
                pass
            else:
                mult = 1.0
                if self.shock_mgr:
                    mult, _ = self.shock_mgr.today_adjustment(day, self.link_id)
                amount = max(0.0, self.base_amount_per_day * mult)
                yield self.pool.put(amount)
            day += 1
            yield self.env.timeout(1.0)

    def request_amount(self, amount):
        yield self.pool.get(amount)

# ========= 场景参数 =========
@dataclass
class Scenario:
    n_units: int = 200
    concrete_per_unit: float = 30
    crane_slots_per_day: float = 240
    workday_hours: int = 8

def build_servers(env, shock_mgr):
    link1 = CapacityServer(env, "link1", 8, lambda: exp_days(0.2), 0.05, shock_mgr)
    link2 = CapacityServer(env, "link2", 6, lambda: exp_days(0.5), 0.03, shock_mgr)
    link3 = MaterialPipeline(env, "link3", 120, shock_mgr, 0.10)
    link4 = CapacityServer(env, "link4", 4, lambda: exp_days(1.0), 0.00, shock_mgr)
    link5 = CapacityServer(env, "link5", 1e9, lambda: 14.0, 0.00, shock_mgr)
    link6 = MaterialPipeline(env, "link6", 240, shock_mgr, 0.0)
    link7 = CapacityServer(env, "link7", 5, lambda: u_int(1,3), 0.0, shock_mgr)
    link8 = CapacityServer(env, "link8", 5, lambda: u_int(1,3), 0.0, shock_mgr)
    return locals()

# ========= 单元施工流程 =========
def unit_process(env, uid, scn, srv, metrics, unit_delay_records):
    start = env.now
    delays = {"waiting_material":0, "waiting_crane":0, "waiting_qa":0, "rework":0}

    if bernoulli(0.2):
        yield env.process(srv["link7"].request_one())
        yield env.process(srv["link8"].request_one())

    t0 = env.now
    yield env.process(srv["link3"].request_amount(scn.concrete_per_unit))
    delays["waiting_material"] += env.now - t0

    t0 = env.now
    yield env.process(srv["link6"].request_amount(1))
    delays["waiting_crane"] += env.now - t0

    passed = False
    while not passed:
        t0 = env.now
        yield env.process(srv["link1"].request_one())
        delays["waiting_qa"] += env.now - t0
        if srv["link1"].fail():
            delays["rework"] += env.now - t0
            yield env.process(srv["link2"].request_one())
        else:
            passed = True

    env.process(srv["link4"].request_one())
    env.process(srv["link5"].request_one())

    finish = env.now
    metrics["unit_finished_times"].append(finish)
    metrics["unit_durations"].append(finish - start)
    for k in delays:
        metrics["delay_breakdown"][k] += delays[k]

    unit_delay_records[uid] = delays

# ========= 扰动示例 =========
def make_default_shocks():
    shocks = []
    for d in range(7, 11):
        shocks.append(Shock(d, "link3", 0.5))
    shocks.append(Shock(15, "link1", 0.5))
    shocks.append(Shock(22, "link1", extra_delay_days=0.5))
    return shocks

# ========= 主程序 =========
def run_baseline(seed=42, scenario=Scenario()):
    RNG.bit_generator.state = np.random.default_rng(seed).bit_generator.state
    random.seed(seed)

    env = simpy.Environment()
    shocks = make_default_shocks()
    shock_mgr = ShockManager(shocks)
    srv = build_servers(env, shock_mgr)

    metrics = dict(
        unit_finished_times=[],
        unit_durations=[],
        wip_ts=[],
        queues={},
        delay_breakdown={"waiting_material":0, "waiting_crane":0, "waiting_qa":0, "rework":0}
    )
    unit_delay_records = {}

    def wip_monitor():
        total = scenario.n_units
        while True:
            day = math.floor(env.now)
            done = len(metrics["unit_finished_times"])
            wip = total - done
            metrics["wip_ts"].append((day, wip))
            yield env.timeout(1.0)
    env.process(wip_monitor())

    for uid in range(scenario.n_units):
        start_day = uid // 10
        def launch(u=uid, sd=start_day):
            yield env.timeout(sd)
            yield env.process(unit_process(env, u, scenario, srv, metrics, unit_delay_records))
        env.process(launch())

    env.run(until=400)

    makespan = max(metrics["unit_finished_times"]) if metrics["unit_finished_times"] else np.nan
    results = {
        "makespan_days": makespan,
        "n_units": scenario.n_units,
        "avg_unit_duration_days": float(np.mean(metrics["unit_durations"])) if metrics["unit_durations"] else np.nan,
    }

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(metrics["wip_ts"], columns=["Day","WIP"]).to_csv("results_wip.csv", index=False)

    gantt_data = []
    finish_times_sorted = sorted(metrics["unit_finished_times"])
    for idx, ft in enumerate(finish_times_sorted):
        gantt_data.append((f"Unit-{idx+1}", 0, ft))
    pd.DataFrame(gantt_data, columns=["Task", "Start", "Finish"]).to_csv("results_gantt.csv", index=False)

    # 输出延误分解（任务粒度）
    delay_rows = []
    for uid, delays in unit_delay_records.items():
        row = {"Task": f"Unit-{uid+1}"}
        row.update(delays)
        delay_rows.append(row)
    pd.DataFrame(delay_rows).to_csv("delay_breakdown.csv", index=False)

    print("=== BASELINE RUN SUMMARY ===")
    for k, v in results.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
    print("延误分解已输出到 delay_breakdown.csv（任务粒度）")

if __name__ == "__main__":
    run_baseline()
