# 📄 Academic Project: Bayesian Logistic Regression for Heart Failure Mortality Prediction

## 🧐 Overview

This project implements a **Bayesian Logistic Regression** model to predict mortality in patients with heart failure using a publicly available dataset. The goal is to estimate the posterior distributions of the model parameters using **Markov Chain Monte Carlo (MCMC)** and quantify the uncertainty in predictions.

Unlike classical logistic regression, the Bayesian approach provides a probabilistic framework that allows us to:

- Model uncertainty in parameter estimates.
- Compute credible intervals for model coefficients.
- Provide probabilistic predictions rather than point estimates.

---

## 💿 Dataset

The project uses a dataset containing clinical records of heart failure patients. Each record includes demographic and clinical features such as age, ejection fraction, serum creatinine, and more. The target variable is:

- `DEATH_EVENT`: Indicates whether the patient died during the follow-up period (`1` = death, `0` = survival).

> **Location:** The primary dataset is located within the `data/` directory.

---

## 🏗️ Project Structure

```
├── data/ # Dataset CSV files, .pkl, etc
├── src/
│ ├── utils/ # Utility functions (MCMC, math, preprocessing)
│ ├── dataset.py # Dataset class for PyTorch
│ ├── model.py # Bayesian Logistic Regression model
│ └── main_train.py # Training script
│ └── main_test.py  # Evaluation script
├── reports/
│ └── figures/ # Posterior plots and visualizations
├── README.md
└── requirements.txt
```

---

## 🚀 Features

The Bayesian Logistic Regression model supports:

- Choice of prior distributions for coefficients:
  - Gaussian
  - Laplace
  - Student-t
- Choice of the normalization tpye:
  - StandardScaler
  - RobustScaler
  - MinMaxScaler
- Optional initialization with classical logistic regression coefficients.
- MCMC sampling via **Metropolis-Hastings**.
- Posterior analysis:
  - Mean, standard deviation, and 95% credible intervals of parameters.
  - Predictive probabilities with uncertainty quantification.
 
---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/alvaroG-IA/Bayesian-Survival-Analysis.git
cd Bayesian-Survival-Analysis
```

2. Create a Python environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---
## ▶️ Usage

The main modules are designed to be executed from the command line using the Python `-m` flag.

### 🔧 Training the Model

The training script initiates an interactive menu prompting the user to select the data normalization method and the prior distribution (Gaussian, Laplace, or Student-t) for MCMC inference.

```bash
# Unix / macOS
python3 -m src.main_train

# Windows
python -m src.main_train
```

---

### 🧪 Testing / Inference

The evaluation script loads the previously trained Bayesian Logistic Regression model, performs predictions, and generates performance metrics and uncertainty analysis.

```bash
# Unix / macOS
python3 -m src.main_test

# Windows
python -m src.main_test
```

---

## 📊 Visualization

Posterior distributions of the model coefficients can be visualized using the
`plot_posterior_distributions` function located in `src/utils/plot_utils.py`.  
These plots provide insight into the uncertainty of each parameter and help assess the stability and reliability of the Bayesian estimates.

These plots, saved in `reports/figures/`, are crucial for:

- Visualizing the shape of the posterior distribution (e.g., Gaussian-like, skewed).

- Assessing the stability of the Bayesian estimates.

- Interpreting the uncertainty associated with each clinical feature.

---

## 📜 License

This project is licensed under the MIT License.

---

## 📨 Contact
Álvaro García Velasco
- e-mail: alvarogarciavelasco1212@gmail.com
- GitHub: [alvaroG-IA](https://github.com/alvaroG-IA)
- LinkedIn: [Álvaro García Velasco](https://www.linkedin.com/in/alvaro-garcia-velasco/)

