# Credit Scoring & Loan Default Prediction

**Author:** John Apollos Olal

**Date:** September 2025

**GitHub:** [JohnApollos](https://github.com/JohnApollos)

### Project Summary

This project addresses a critical business problem in the fintech sector: credit risk assessment. The goal was to build a machine learning model to predict loan default probability for a microfinance institution. Using a loan prediction dataset from Kaggle, the project involved a complete data science workflow, including data cleaning, imputation of missing values, and feature engineering to prepare the data for modeling. A **Logistic Regression** model was trained and evaluated, achieving an overall **accuracy of 79%**. More importantly, the model demonstrated a **precision of 95%** in predicting defaults, indicating high reliability when flagging high-risk applicants, though its recall of 42% suggests opportunities for future improvement in identifying all potential defaulters.

### Key Business Problem

Microfinance institutions and digital lenders need to accurately assess the risk of a new applicant defaulting on a loan. A reliable predictive model can significantly reduce financial losses by flagging high-risk applicants before a loan is disbursed.

### Model Performance

The final Logistic Regression model was evaluated on a held-out test set (20% of the original data).

* **Overall Accuracy:** **79%**

* **Confusion Matrix:**

    ```
    [[18 25]
     [ 1 79]]
    ```

    * **True Negatives (Correctly predicted defaults):** 18
    * **True Positives (Correctly predicted repayments):** 79
    * **False Negatives (Risky loans approved):** 1
    * **False Positives (Good loans rejected):** 25

* **Classification Report Insights:**
    * **Precision (for Defaults): 95%** — When the model predicts an applicant will default, it is correct 95% of the time.
    * **Recall (for Defaults): 42%** — The model successfully identifies 42% of all applicants who will actually default.

### Technical Skills & Tools

* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-learn
    * **Model:** Logistic Regression
    * **Metrics:** Accuracy, Confusion Matrix, Classification Report
    * **Preprocessing:** `train_test_split`, One-Hot Encoding
* **Environment:** Jupyter Notebook
* **Programming Language:** Python

### Data Source

* **Dataset:** [Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) on Kaggle.

### Project Structure

```
credit-scoring-model/
│
├── train_loan.csv         # The raw training data
├── test_loan.csv          # The raw test data for final predictions
├── Credit_Score_Model.ipynb # The Jupyter Notebook with all code and analysis
└── README.md              # This file
```
### How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/JohnApollos/credit-scoring-model.git](https://github.com/JohnApollos/credit-scoring-model.git)
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd credit-scoring-model
    ```
3.  **Install the required libraries:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
4.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
5.  Open the `Credit_Score_Model.ipynb` file and run the cells.
```eof