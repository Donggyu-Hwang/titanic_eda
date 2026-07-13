# Titanic EDA

## Streamlit Cloud diagnostic kit

Use these files as the Streamlit Cloud Main file path, in order, to isolate the crash point without changing the code again:

1. `diagnostics/00_streamlit_boot.py`
1. `diagnostics/01_core_imports.py`
1. `diagnostics/02_plotting_imports.py`
1. `diagnostics/03_local_data.py`
1. `diagnostics/04_model_train.py`
1. `diagnostics/05_full_smoke.py`

If a stage crashes, the issue is in that stage or earlier. If a stage loads, move to the next one.
