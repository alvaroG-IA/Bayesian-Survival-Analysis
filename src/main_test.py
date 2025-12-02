from pathlib import Path
from src.data import HeartHealthDataset
from src.utils.persistence import load_pipeline
from sklearn.metrics import classification_report

# ============================
#   DEFINICIÓN DE RUTAS
# ============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PIPELINE_PATH = str(DATA_DIR / "full_model_pipeline.pkl")
CSV_TEST_PATH = str(DATA_DIR / "fallo_cardiaco_test.csv")

# ============================
#   CARGA DEL PIPELINE
# ============================
pipeline = load_pipeline(PIPELINE_PATH)
if pipeline is None:
    exit()

# ============================
#   INFORMACIÓN DEL PIPELINE
# ============================
prior_dict = {
    1: 'Normal',
    2: 'Laplace',
    3: 'Student-t'
}

print("\n=== Información del modelo cargado ===")
print(f"Scaler usado: {pipeline.scaler_name}")
print("Prior usado:", prior_dict[pipeline.prior_func_used])
print(f"Iteraciones: {pipeline.iterations}")
print(f"Burn-in: {pipeline.burn_in*100:.1f}%")
print(f"Proposal width de la distribución de propuesta: {pipeline.proposal_width}\n")

# ============================
#   CARGA DE DATOS DE TEST
# ============================
dataset = HeartHealthDataset(CSV_TEST_PATH)
X, y = dataset.get_raw_data()

# ============================
#   PREDICCIÓN
# ============================
preds = pipeline.predict(X)

# ============================
#   MÉTRICAS
# ============================
print(classification_report(y, preds))
