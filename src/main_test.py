from pathlib import Path
from src.data import HeartHealthDataset
from src.utils.persistence import load_pipeline
from sklearn.metrics import classification_report

# ============================
#       ROOTS DEFINITION
# ============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PIPELINE_PATH = str(DATA_DIR / "full_model_pipeline.pkl")
CSV_TEST_PATH = str(DATA_DIR / "fallo_cardiaco_test.csv")

# ============================
#        PIPELINE LOAD
# ============================
pipeline = load_pipeline(PIPELINE_PATH)
if pipeline is None:
    exit()

# ============================
#     PIPELINE INFORMATION
# ============================
prior_dict = {
    1: 'Normal',
    2: 'Laplace',
    3: 'Student-t'
}

print("\n=== Information of the pipeline loaded ===")
print(f"Scaler used: {pipeline.scaler_name}")
print("Prior-func used:", prior_dict[pipeline.prior_func_used])
print(f"Num iterations: {pipeline.iterations}")
print(f"Burn-in: {pipeline.burn_in*100:.1f}%")
print(f"Proposal width of the propose function: {pipeline.proposal_width}\n")

# ============================
#         DATA LOAD
# ============================
dataset = HeartHealthDataset(CSV_TEST_PATH)
X, y = dataset.get_raw_data()

# ============================
#         PREDICTION
# ============================
preds = pipeline.predict(X)

# ============================
#           METRICS
# ============================
print(classification_report(y, preds))
