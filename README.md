# ChurnScope: Predictive Modeling for Telecom Retention

## 📊 Project Overview
This project develops a machine learning pipeline to identify high-risk customers within a telecommunications dataset. By leveraging demographic data, account information, and service usage patterns, the project compares three distinct modeling approaches to determine the most effective method for predicting customer churn.

##  Tech Stack
* **Language:** R (RMarkdown)
* **Key Libraries:**
    * `tidyverse`: Data manipulation and visualization.
    * `caret`: Training and cross-validation framework.
    * `recipes`: Feature engineering and preprocessing.
    * `xgboost` & `randomForest`: Advanced ensemble modeling.
    * `pROC`: Model evaluation and AUC metrics.

##  Data Processing & Engineering
The pipeline includes a robust cleaning and engineering phase:
* **Imputation:** Logical imputation of `total_charges` based on tenure and monthly rates.
* **Feature Engineering:**
    * **Tenure Binning:** Grouping customers into temporal cohorts (e.g., 0-6 months, 49+ months).
    * **Service Aggregation:** A `total_services` count derived from 7 different service columns.
    * **Financial Indicators:** Discretizing monthly charges into high/low categories based on median values.
* **Class Imbalance:** Applied **up-sampling** to the minority class (`Churn = Yes`) to ensure models are not biased toward the majority class.
* **Scaling & Encoding:** One-hot encoding for categorical variables and centering/scaling for numeric predictors via the `recipes` package.

##  Models Evaluated
The project implements and compares three distinct algorithms:
1. **Logistic Regression:** A baseline probabilistic model.
2. **Random Forest:** A non-linear ensemble approach using 400 trees with importance tracking.
3. **XGBoost:** Gradient boosted decision trees optimized for AUC performance.
Scaling & Encoding: One-hot encoding for categorical variables and centering/scaling for numeric predictors.
