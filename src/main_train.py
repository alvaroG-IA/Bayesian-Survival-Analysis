import pandas as pd
from pathlib import Path
from src.data import HeartHealthDataset
from src.models import LogisticBayesModel
from src.utils.helpers import (
    set_seed,
    plot_post_distribuitions,
    seleccionar_preprocesado,
    seleccionar_prior_func,
)
from src.utils.persistence import save_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# ============================
#     FIJADO DE SEMILLA
# ============================
SEED = 42
set_seed(SEED)

# ============================
#     DEFINICIÓN DE RUTAS
# ============================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PIPELINE_PATH = str(DATA_DIR / "full_model_pipeline.pkl")
CSV_PATH = str(DATA_DIR / "fallo_cardiaco.csv")
CSV_TRAIN_PATH = str(DATA_DIR / "fallo_cardiaco_train.csv")
CSV_TEST_PATH = str(DATA_DIR / "fallo_cardiaco_test.csv")


# ============================
#       CARGA DE DATOS
# ============================
df = pd.read_csv(CSV_PATH)
labels = df['DEATH_EVENT']
data = df.drop(['DEATH_EVENT'], axis=1)
col_names = data.columns.values

continuous_cols = [
    "age",
    "creatinine_phosphokinase",
    "ejection_fraction",
    "platelets",
    "serum_creatinine",
    "serum_sodium",
    "time"
]

binary_cols = [
    "anaemia",
    "diabetes",
    "high_blood_pressure",
    "sex",
    "smoking"
]

# ============================
#       TRAIN/TEST SPLIT
# ============================
data_train, data_test, labels_train, labels_test = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=SEED
)

pd.concat([data_train, labels_train], axis=1).to_csv(CSV_TRAIN_PATH, index=False)
pd.concat([data_test, labels_test], axis=1).to_csv(CSV_TEST_PATH, index=False)


dataset = HeartHealthDataset(CSV_TRAIN_PATH)
X, y = dataset.get_raw_data()

# ============================
#     SELECCIÓN DEL SCALER
# ============================
scaler_opt = seleccionar_preprocesado()

if scaler_opt == 1:
    scaler = StandardScaler()
elif scaler_opt == 2:
    scaler = RobustScaler()
else:
    scaler = MinMaxScaler()


# =================================
#   SELECCIÓN DE LA FUNCIÓN PRIOR
# =================================
prior_func_opt = seleccionar_prior_func()


# ============================
#   COLUMN-TRANSFORMER FINAL
# ============================
preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, continuous_cols),
        ('bin', 'passthrough', binary_cols)
    ]
)

# ============================
#      PIPELINE GENERAL
# ============================
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LogisticBayesModel(prior_func_opt=prior_func_opt))
])

# ============================
#       ENTRENAMIENTO
# ============================
pipeline['preprocessor'].fit(X)
X_transformed = pipeline['preprocessor'].transform(X)

iterations = 10000
burn_in = 0.05
proposal_width = 0.15

pipeline['model'].fit(
    X_transformed, y,
    iterations=iterations,
    burn_in=burn_in,
    proposal_width=proposal_width
)

# ============================
#         RESULTADOS
# ============================
model_fitted = pipeline['model']
print(f'Ratio de aceptación -> {model_fitted.acceptance_ratio:.2f}%')

pipeline.scaler_name = type(scaler).__name__
pipeline.prior_func_used = prior_func_opt
pipeline.iterations = iterations
pipeline.burn_in = burn_in
pipeline.proposal_width = proposal_width

samples = model_fitted.samples_
w_mean = model_fitted.w_mean_
w_std = model_fitted.w_std_
w_ci = model_fitted.w_ci_

# ===========================================
#   NOMBRES DE VARIABLES POST-PROCESAMIENTO
# ===========================================
all_features = ["intercept"] + continuous_cols + binary_cols    # importante mantener este orden tras uso de ColumTransformer

# Construimos DataFrame con coeficientes y estadísticas
coef_data = {
    "feature": all_features,
    "w_mean": w_mean,
    "w_std": w_std,
    "ci_lower": w_ci[0],
    "ci_upper": w_ci[1]
}

df_coef = pd.DataFrame(coef_data)

# Quitamos el intercepto para ver solo las variables
df_features = df_coef[df_coef["feature"] != "intercept"].copy()

# Magnitud del peso para ordenar por importancia
df_features["abs_w"] = df_features["w_mean"].abs()

# Ordenamos
df_sorted = df_features.sort_values(by="abs_w", ascending=False)

# Top N
TOP_N = 5
df_top = df_sorted.head(TOP_N)

print(f"=== TOP {TOP_N} FEATURES POR IMPORTANCIA ===")

for _, row in df_top.iterrows():
    print(f'Feature: {row["feature"]}')
    print(f' * w_mean = {row["w_mean"]:.4f}')
    print(f' * w_std = {row["w_std"]:.4f}')
    print(f' * CI95 = ({row["ci_lower"]:.4f}, {row["ci_upper"]:.4f})')
    print("")

plot_post_distribuitions(samples=samples[:, 1:], w_mean=w_mean[1:], w_std=w_std[1:], n_dim=len(all_features) - 1,
                         scaler_opt=scaler_opt, prior_func_opt=prior_func_opt, col_names=all_features[1:],
                         save_dir=str(PROJECT_ROOT / 'reports' / 'figures'))

# ============================
#   GUARDADO DEL PIPELINE
# ============================
save_pipeline(pipeline, PIPELINE_PATH)
