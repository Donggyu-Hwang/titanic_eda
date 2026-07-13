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

## How to read the result

- If `00_streamlit_boot.py` crashes on refresh, the problem is below your app code and points to the Streamlit Cloud runtime or wrapper process.
- If `00` works but `01` crashes, the issue is in the core Python import stack.
- If `01` works but `02` crashes, the issue is likely in matplotlib or seaborn.
- If `02` works but `03` crashes, the issue is local file access or the dataset path.
- If `03` works but `04` crashes, the issue is the scikit-learn training stack.
- If only `05` crashes, the issue is in the full combination of imports, plotting, and model training.
- If all stages pass but `streamlit_app.py` still crashes, the remaining problem is in the main app logic or its rerun behavior.
