import shutil
from pathlib import Path

from partA import Scenario, run_baseline


def main() -> None:
    run_baseline(seed=7, scenario=Scenario(n_units=20))
    generated = [Path("results_gantt.csv"), Path("results_wip.csv"), Path("delay_breakdown.csv")]
    for path in generated:
        assert path.exists()
        path.unlink()
    shutil.rmtree("results", ignore_errors=True)
    shutil.rmtree("__pycache__", ignore_errors=True)
    print("llm-simpy-project-scheduler smoke test passed")


if __name__ == "__main__":
    main()
