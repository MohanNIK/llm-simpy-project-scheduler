# llm-simpy-project-scheduler

`llm-simpy-project-scheduler` is a publishable snapshot of a research prototype for project scheduling simulation with optional LLM-guided decisions. It combines a deterministic SimPy baseline, an enhanced scheduling model, and a Streamlit interface for scenario exploration.

## Scope

- Baseline scheduling simulation in `partA.py`
- Enhanced simulation and critical-path reporting in `partB.py`
- Optional LLM policy wrapper in `partC-llm.py` and `policies/llm_policy.py`
- Local UI in `streamlit_app.py`
- Sample baseline CSV outputs in `examples/`

## Quick Start

```powershell
python -m pip install -r requirements.txt
python smoke_test.py
streamlit run streamlit_app.py
```

Without an API key, the LLM policy safely falls back to `noop` decisions. This public repo does not include any bundled key.

## Included Sample Outputs

- `examples/results_gantt.csv`
- `examples/results_wip.csv`
- `examples/delay_breakdown.csv`
- `docs/assets/baseline_wip.png`

## Notes

- This repo excludes local virtual environments and result caches.
- Live LLM calls require `DASHSCOPE_API_KEY` or `QWEN_API_KEY`.
- The sample outputs are baseline artifacts for documentation and UI sanity checks.
