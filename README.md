# 📊 Pima Diabetes Risk Prediction & Actuarial Pricing (`CMP-N205-0`)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![Data Science](https://img.shields.io/badge/Domain-Data%20Science%20%26%20Healthcare-blue?style=flat)
![ML Model](https://img.shields.io/badge/ML-Gaussian%20Naive%20Bayes-orange?style=flat)
![Dataset](https://img.shields.io/badge/Dataset-Pima%20Indians%20Diabetes-green?style=flat)
![Academic](https://img.shields.io/badge/Academic-Roehampton-red?style=flat)
![License](https://img.shields.io/badge/License-Academic%20Use-lightgrey?style=flat)

An end-to-end data science study analyzing biophysical markers from the **Pima Indians Diabetes Database** to predict type-2 diabetes risk. Formulated from the perspective of a fictional UK insurance firm (**HealthSure UK Ltd**), the project combines biological pre-processing, median imputation, outlier capping, and probabilistic classification via **Gaussian Naive Bayes** to support early clinical intervention and actuarial risk segmentation.

---

## 📌 Project Information

- **Author:** Bhowan Khawas
- **Student ID:** A00028871
- **Program:** BSc Computing
- **Module:** CMP-N205-0 Data Science
- **Institution:** University of Roehampton London
- **Submission Date:** 17 April 2026

---

## 🎯 Business Context & Hypotheses

Undiagnosed type-2 diabetes introduces significant long-term financial liabilities for health insurers due to severe secondary complications (renal failure, cardiovascular disease, neuropathy). By screening policyholders using biophysical data, **HealthSure UK Ltd** aims to establish proactive intervention protocols and personalized premium pricing.

### Key Business Problems
1. **Risk Prediction for Early Intervention:** Can biometric markers (Glucose, BMI, Age, Insulin) accurately identify undiagnosed diabetic individuals to offer pre-emptive GP consultations?
2. **Premium Segmentation & Actuarial Pricing:** Can policyholders be classified into evidence-based risk tiers (**Low**, **Medium**, **High**) for differential premium calculations?
3. **Strongest Clinical Predictors:** Which biological markers exhibit the strongest conditional probability distributions, allowing HealthSure to streamline initial policy questionnaires?

---

## 📊 Dataset & Pre-processing Pipeline

The project utilizes the **Pima Indians Diabetes Database** (NIDDK / Kaggle), containing **768 female patient observations** across **8 biometric predictors** and **1 binary target variable** (`Outcome`).

### Feature Overview
- **Predictors:** `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`.
- **Target:** `Outcome` (`1` = Diabetic, `0` = Non-Diabetic).

### Data Cleaning & Pre-processing Summary

Physiologically impossible zero values (e.g., zero blood pressure or zero glucose) were identified as hidden missing values and systematically pre-processed:

| Step | Action | Columns Affected | Description |
| :---: | :--- | :--- | :--- |
| **1** | **Identify Hidden Missing Data** | `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` | Replaced biologically impossible `0` values with `NaN`. |
| **2** | **Quantify Impact** | All 5 biometric columns | 376 rows (49%) contained at least one impossible zero. |
| **3** | **Median Imputation** | `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI` | Imputed medians: Glucose=117, BP=72, Skin=29, Insulin=125, BMI=32.3. |
| **4** | **Duplicate Verification** | Entire Dataset | Confirmed 0 duplicate rows across all 768 records. |
| **5** | **Outlier Capping (IQR)** | `Insulin` | Capped extreme right-skewed values above upper fence (135.9). |
| **6** | **Type Verification** | Entire Dataset | Verified all variables as numeric (`int64` / `float64`). |

---

## 🤖 Machine Learning Technique: Gaussian Naive Bayes

The core modeling strategy utilizes a **Gaussian Naive Bayes Classifier** to compute posterior class probabilities:

$$P(C_k \mid X) \propto P(C_k) \prod_{i=1}^{n} P(x_i \mid C_k)$$

Where continuous features are modeled using normal distributions parameterized by class-specific means ($\mu_k$) and variances ($\sigma_k^2$).

### Model Justification Matrix

| Criteria | Justification for Gaussian Naive Bayes |
| :--- | :--- |
| **Binary Output** | Naturally produces calibrated class probabilities $P(\text{Diabetic} \mid X)$ needed for risk scoring. |
| **Continuous Features** | Accurately models Gaussian biometric features post-imputation and capping. |
| **Regulatory Compliance (XAI)** | Highly interpretable parameters ($P(x_i \mid C_k)$) satisfy UK insurance explainability standards. |
| **Sample Efficiency** | Requires estimating only $2 \times n$ parameters, preventing overfitting on $N=768$ samples. |
| **Alternative Rejections** | **K-Means** rejected (unsupervised), **ARM** rejected (requires discretization), **KNN** rejected (sensitive to scaling and heavy imputation). |

---

## 📂 Repository Structure

.
├── Coursework1_Report.pdf     # Comprehensive Academic Report (Data Discussion & ML Formulation)
├── diabetes.csv               # Raw Pima Indians Diabetes Dataset (Kaggle)
├── diabetes_cleaned.csv       # Pre-processed dataset (Median Imputed & IQR Capped)
└── README.md                  # Repository documentation


---

## 📚 References

1. **UCI Machine Learning Repository / Kaggle (2016):** *Pima Indians Diabetes Database*. [Kaggle Dataset Link](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
2. **Smith, J.W., et al. (1988):** *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*. Proc. Annu. Symp. Comput. Appl. Med. Care, pp. 261–265.
3. **Mitchell, T. (1997):** *Machine Learning*. McGraw-Hill. Chapter 6: Bayesian Learning.
4. **Han, J., Kamber, M., & Pei, J. (2011):** *Data Mining: Concepts and Techniques*, 3rd edn. Morgan Kaufmann.
5. **McKinney, W. (2017):** *Python for Data Analysis*, 2nd edn. O'Reilly Media.
