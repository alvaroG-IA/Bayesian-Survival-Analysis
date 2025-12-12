# Bayesian Logistic Regression for Heart Failure Mortality Prediction

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

> Note: The dataset can be found in the `data/` directory.

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

To run the project, you can execute the training and testing modules directly from the command line using Python's `-m` flag.

### 🔧 Training the Model

The training script is located in `src/main_train`.  
You can run it using:

#### **Unix / macOS**
```bash
python3 -m src.main_train
```
#### **Windows**
```bash
python -m src.main_train
```
When the training script starts, an interactive menu will appear.
You will be prompted to select:

1. The type of data normalization (StandardScaler, RobustScaler, or MinMaxScaler).
2. The prior distribution to be used in the Bayesian Logistic Regression model (Gaussian, Laplace, or Student-t).

These selections determine how the data will be preprocessed and which prior will be used for parameter sampling during MCMC inference.

---

### 🧪 Testing / Inference
To run the model evaluation or generate predictions, execute the testing script `src/main_test`:

#### **Unix / macOS**
```bash
python3 -m src.main_test
```
#### **Windows**
```bash
python -m src.main_test
```
The testing script loads the previously trained Bayesian Logistic Regression model and evaluates it on the corresponding dataset, producing predictions and performance metrics.

---

## 📊 Visualization

Posterior distributions of the model coefficients can be visualized using the
`plot_posterior_distributions` function located in `src/utils/plot_utils.py`.  
These plots provide insight into the uncertainty of each parameter and help assess the stability and reliability of the Bayesian estimates.

During the execution of the training script, the posterior distribution plots are automatically generated and saved in the `reports/figures/` directory.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
