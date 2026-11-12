https://datascience-week2---weintern-project-enqfqd4op5pfkf2d4ifidj.streamlit.app/
# DataScience-Week2---WeIntern-Project
## Rossmann Store Sales: Data Cleaning, Exploratory Data Analysis & Sales Prediction

## 1. Project Overview

This project is part of the **WeIntern Week 2 Data Science Internship Assignment**.

The project uses the Rossmann Store Sales dataset to perform an end-to-end data science workflow, covering data preprocessing, exploratory analysis, and machine learning-based sales prediction.

The assignment is divided into three tasks:

| Task | Title | Main Objective |
|---|---|---|
| Task 1 | Data Cleaning using Pandas | Clean, validate, and prepare datasets. |
| Task 2 | Exploratory Data Analysis (EDA) | Analyze sales patterns and discover business insights. |
| Task 3 | Sales Prediction | Build and evaluate machine learning models to predict sales. |

Each task is implemented in a separate Jupyter Notebook.

---

## 2. Dataset Description

The project uses the **Rossmann Store Sales dataset**, which contains historical retail sales information and store-related attributes.

### Datasets Used

| File | Description | Purpose |
|---|---|---|
| `train.csv` | Historical sales records, including Sales and Customers. | Data cleaning, EDA, and model training. |
| `test.csv` | Store-level daily records without actual Sales values. | Data cleaning and final sales prediction. |
| `store.csv` | Store information, including StoreType, Assortment, competition, and promotion details. | Data cleaning, merging, EDA, and feature engineering. |
| `sample_submission.csv` | Reference submission file containing Id and Sales columns. | Final prediction format validation. |

### Dataset Characteristics

- Training dataset: 1,017,209 records and 9 columns.
- Testing dataset: 41,088 records and 8 columns.
- Store dataset: 1,115 records and 10 original columns.
- Sample submission: 41,088 records and 2 columns.

**Important considerations:**

- `Sales` is the target variable.
- `Customers` is available in the training data but not in the test data. Therefore, it is excluded from model features.
- The test dataset is not used for model training or validation.
- `sample_submission.csv` is used to maintain the required prediction structure.

---

# 3. Task 1 – Data Cleaning using Pandas

### Objective

To inspect, clean, validate, and prepare the raw Rossmann datasets for further analysis and machine learning.

### Work Performed

**1. Data Loading and Inspection**
- Loaded all four CSV datasets.
- Inspected dataset dimensions and column information.
- Checked data types and statistical summaries.
- Created separate raw and cleaned dataset copies.

**2. Missing Value Analysis**
- Identified missing values in each dataset.
- Calculated missing-value counts and percentages.
- Visualized missing values.
- Applied appropriate missing-value handling techniques.

**3. Duplicate Handling**
- Identified duplicate records.
- Checked duplicate store records.
- Removed duplicate records where appropriate.

**4. Data Standardization**
- Standardized column names.
- Corrected data types.
- Standardized date and categorical columns.

**5. Business Validation**
- Checked invalid and inconsistent values.
- Validated sales, customer, store, and promotion-related information.
- Checked business rules and dataset consistency.

**6. Missing Value Treatment**
- Handled missing store-related information.
- Used median imputation for `CompetitionDistance`.
- Preserved indicators for missing competition information.
- Handled structurally missing Promo2-related information appropriately.
- Filled missing `Open` values in the test dataset using historical store and weekday information, with a global fallback.

**7. Outlier Analysis**
- Applied the Interquartile Range (IQR) method.
- Identified potential outliers in Sales and Customers.
- Examined outliers in open-store records.
- Retained genuine business outliers instead of blindly deleting or capping them.

**8. Final Validation and Export**
- Performed final data quality checks.
- Compared datasets before and after cleaning.
- Generated cleaning summaries.
- Exported cleaned training, testing, and store datasets.

### Task 1 Deliverables

- Cleaned training dataset.
- Cleaned testing dataset.
- Cleaned store dataset.
- Missing-value analysis.
- Duplicate analysis.
- Outlier analysis.
- Before-and-after comparison.
- Final validation summary.
- Cleaning visualizations.
- Data Cleaning Report (PDF and TXT).

---

# 4. Task 2 – Exploratory Data Analysis (EDA)

### Objective

To perform exploratory data analysis on the cleaned Rossmann dataset and identify sales patterns, trends, relationships, and business insights.

### Work Performed

**1. Sales and Customer Analysis**
- Analyzed sales distribution.
- Analyzed customer distribution.
- Examined sales outliers.
- Studied the relationship between sales and customers.

**2. Store Analysis**
- Analyzed store type distribution.
- Examined assortment categories.
- Compared sales across different store types.
- Compared sales across assortment categories.

**3. Promotion Analysis**
- Analyzed promotional and non-promotional sales.
- Examined the relationship between promotions and sales.

**4. Correlation Analysis**
- Examined relationships between numerical variables.
- Generated a correlation heatmap.

**5. Time-Based Analysis**
- Analyzed daily sales trends.
- Examined monthly sales.
- Compared yearly sales.
- Analyzed sales across weekdays.
- Compared monthly sales across different years.

**6. Holiday Analysis**
- Examined sales during school holidays.
- Analyzed sales during state holidays.

**7. Store Performance Analysis**
- Identified the top 10 performing stores.
- Identified the bottom 10 performing stores.

### EDA Visualizations

A total of 20 visualizations were created:

| No. | Visualization |
|---:|---|
| 1 | Sales Distribution |
| 2 | Customers Distribution |
| 3 | Sales Boxplot |
| 4 | Store Type Distribution |
| 5 | Assortment Distribution |
| 6 | Promotion Distribution |
| 7 | Sales vs Customers |
| 8 | Sales vs Promotion |
| 9 | Sales by Store Type |
| 10 | Sales by Assortment |
| 11 | Correlation Heatmap |
| 12 | Daily Sales Trend |
| 13 | Monthly Sales |
| 14 | Yearly Sales |
| 15 | Weekday Sales |
| 16 | Monthly Sales by Year |
| 17 | Sales by School Holiday |
| 18 | Sales by State Holiday |
| 19 | Top 10 Stores |
| 20 | Bottom 10 Stores |

### Task 2 Deliverables

- 20 EDA visualizations.
- Sales distribution analysis.
- Store performance analysis.
- Promotion and holiday analysis.
- Time-based sales analysis.
- Business insights.
- EDA Summary Report (PDF and TXT).

---

# 5. Task 3 – Sales Prediction using Machine Learning

### Objective

To build and evaluate machine learning regression models that predict Rossmann store sales using historical sales data and relevant store-related features.

### Work Performed

**1. Data Preparation**
- Loaded cleaned training, testing, and store datasets.
- Merged sales data with store information.
- Prepared training and testing datasets for modeling.

**2. Feature Engineering**
- Extracted date-related features.
- Created year, month, day, and weekday features.
- Prepared promotion-related features.
- Created competition-related features.
- Prepared relevant numerical and categorical variables.
- Generated a model-ready feature set containing 27 features.

**3. Data Leakage Prevention**
- Excluded `Customers` because it is unavailable in the test dataset.
- Excluded `Id` and raw `Date` from model features.
- Used chronological splitting instead of random splitting.
- Used open-store records for model training.

**4. Training and Validation**
- Selected 844,392 open-store records for training and validation.
- Split the data chronologically.
- Used a training period ending on January 24, 2015, for the training portion.
- Evaluated models on the subsequent validation period.

**5. Model Development**

The following regression models were implemented:

- Linear Regression
- Random Forest Regressor

**6. Model Evaluation**

Models were evaluated using:

- RMSE – Root Mean Squared Error
- MAE – Mean Absolute Error
- R² Score

### Model Performance

| Model | RMSE | MAE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 2664.04 | 1946.24 | 0.24 |
| Random Forest Regressor | 1363.03 | 928.15 | 0.80 |

**Result:** Random Forest performed better than Linear Regression on the reported validation results.

**Note:** The Random Forest evaluation used a 150,000-record sample.

**7. Hyperparameter Tuning**

RandomizedSearchCV was used with TimeSeriesSplit for hyperparameter tuning.

Best parameters obtained:

| Parameter | Best Value |
|---|---|
| `n_estimators` | 80 |
| `min_samples_split` | 5 |
| `min_samples_leaf` | 2 |
| `max_features` | 0.8 |
| `max_depth` | 25 |

Best cross-validation RMSE: **2175.26**

The tuning process used a separate 20,000-record sample. Therefore, its cross-validation RMSE should not be directly compared with the full validation results reported above.

**8. Final Prediction**
- Prepared the final test dataset.
- Generated sales predictions.
- Handled closed-store predictions appropriately.
- Maintained the required `Id` and `Sales` submission structure.

### Task 3 Deliverables

- Trained machine learning model.
- Model comparison results.
- Validation predictions.
- Final test predictions.
- Feature importance analysis.
- Model evaluation charts.
- Sales Prediction Report (PDF and TXT).

---

## 6. Project Directory Structure

```text
DataScience-Week2---WeIntern-Project/
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── store.csv
│   │   └── sample_submission.csv
│   │
│   └── cleaned/
│       ├── train_cleaned.csv
│       ├── test_cleaned.csv
│       └── store_cleaned.csv
│
├── notebooks/
│   ├── Task_01_Data_Cleaning.ipynb
│   ├── Task_02_EDA.ipynb
│   └── Task_03_Sales_Prediction.ipynb
│
├── outputs/
│   ├── charts/
│   │   ├── Task1/--.png
│   │   ├── Task2/--.png
│   │   └── Task3/---.png
│   │
│   ├── models/---.pkl
│   ├── predictions/---.csv
│
├── reports/
│   ├── Data_Cleaning_Report.pdf
│   ├── Data_Cleaning_Report.txt
│   ├── EDA_Summary_Report.pdf
│   ├── EDA_Summary_Report.txt
│   ├── Sales_Prediction_Report.pdf
│   └── Sales_Prediction_Report.txt
│
└── README.md
```
---
## 7. Technologies and Libraries
| Technology | Application |
|---|---|
| Python | Programming language |
| Pandas | Data cleaning and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Machine learning and model evaluation |
| Joblib / Pickle | Model serialization |
| Google Colab | Notebook development and execution |
| Git | Version control |
| GitHub | Project hosting and documentation |
---
## 8. How to Run the Project
### Step 1 – Clone the Repository
```bash
git clone https://github.com/aadeshbhamare/DataScience-Week2---WeIntern-Project.git
```
### Step 2 – Navigate to the Project
```bash
cd DataScience-Week2---WeIntern-Project
```
### Step 3 – Install Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```
### Step 4 – Place the Raw Datasets
Place the four original CSV files inside:
```text
data/raw/
```
### Step 5 – Run the Notebooks in Sequence
Execute the notebooks in the following order:
1. `01_Data_Cleaning.ipynb`
2. `02_Exploratory_Data_Analysis.ipynb`
3. `03_Sales_Prediction.ipynb`
**Execution Note:** Task 1 generates the cleaned datasets required by Task 2 and Task 3. Therefore, execute the notebooks in the specified order and ensure the required data paths are available.
---
## 9. Key Skills Demonstrated
- Python Programming
- Data Cleaning and Preprocessing
- Missing Value Treatment
- Duplicate Detection and Handling
- Data Validation
- Outlier Detection using IQR
- Exploratory Data Analysis
- Statistical Analysis
- Data Visualization
- Feature Engineering
- Regression Analysis
- Chronological Train-Validation Splitting
- Random Forest Regression
- Hyperparameter Tuning
- Model Evaluation
- Sales Prediction
- Git and GitHub
---
## 10. Conclusion
This project demonstrates a complete data science workflow using the Rossmann Store Sales dataset.
The work begins with cleaning and validating raw datasets, progresses through exploratory data analysis to identify meaningful sales patterns, and concludes with machine learning models for sales prediction.
The project provides practical experience in handling real-world retail data, performing business-oriented analysis, developing predictive models, and organizing project deliverables professionally.
---
## 11. Author
**Aadesh Bhamare**

**Project:** WeIntern – Week 2 Data Science Internship
