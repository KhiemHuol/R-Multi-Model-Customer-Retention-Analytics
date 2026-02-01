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

##  Model Outputs & Key Metrics

### Dataset Split (Test Set)
* **Non-Churn (No):** 1,034
* **Churn (Yes):** 373

### Performance Comparison (Threshold = 0.5)
The following table summarizes the performance of each model after training on up-sampled data:

| Model | Accuracy | Sensitivity (TPR) | Specificity | AUC |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.7420 | **0.7641** | 0.7340 | 0.8479 |
| **Random Forest** | **0.7910** | 0.6434 | **0.8443** | **0.8504** |
| **XGBoost** | 0.7719 | 0.6944 | 0.7998 | 0.8451 |



### Detailed Confusion Matrices
* **Logistic Regression:** TN=759, FN=88, FP=275, TP=285
* **Random Forest:** TN=873, FN=133, FP=161, TP=240
* **XGBoost:** TN=827, FN=114, FP=207, TP=259

---

##  Key Insights & Interpretation
* **Discrimination Power:** All models showed strong discrimination with **AUC values between 0.845 and 0.850**.
* **The Accuracy vs. Sensitivity Trade-off:** * **Random Forest** achieved the highest overall accuracy (**79.1%**) and the best AUC (**0.8504**), making it the best model for minimizing false alarms.
    * **Logistic Regression** provided the highest sensitivity (**0.7641**), making it the most effective tool if the business goal is to catch as many potential churners as possible, despite a higher rate of false positives.

##  Limitations & Challenges
* **Feature Constraints:** The dataset lacks behavioral usage metrics such as call logs, data consumption, and billing history.
* **Missing Data:** The `churn_reason` field was missing for the majority of the dataset (5,174 out of 7,043 records).
* **Threshold Optimization:** A default classification threshold of **0.5** was used; this may not be optimized for specific business costs.

## ✅ Conclusion
The models effectively address the problem by providing reliable rankings of customers by churn risk. While accuracy sits just under 80%, the high AUC indicates the model is useful for a retention program.
* **Priority - Recall:** Use Logistic Regression to catch the most churners.
* **Priority - Precision:** Use Random Forest to minimize false positives and maximize accuracy.
