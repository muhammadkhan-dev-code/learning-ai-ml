# Student Math Performance Analysis using Machine Learning

## Project Overview

This project analyzes the **Student Math Performance Dataset** to understand the factors affecting students' academic performance. The dataset was preprocessed and explored using **Exploratory Data Analysis (EDA)** before applying classification algorithms.

The objective is to predict whether a student will **Pass** or **Fail** based on demographic, family, social, lifestyle, and academic features.

---

## Problem Type

**Binary Classification**

The original target variable (`G3`) contains the final mathematics grade (0–20). It was converted into a binary target variable:

- **Pass (1):** G3 >= 10
- **Fail (0):** G3 < 10

The new target variable is named **Result**.

---

# 1. Data Loading

The Student Math Performance dataset was loaded into a Pandas DataFrame.

The dataset was initially inspected using:

- `head()`
- `tail()`
- `shape`
- `info()`
- `describe()`

These functions were used to understand the dataset structure, dimensions, data types, and basic statistical properties.

### Dataset Summary

- **Total Students:** 395
- **Total Features:** 33
- **Classification Target:** Result

---

# 2. Data Cleaning

The following data-cleaning and preparation steps were performed:

- Removed the unnecessary `Unnamed: 0` index column.
- Checked for missing values.
- Checked for duplicate records.
- Verified data types.
- Created the binary target variable (`Result`).
- Separated categorical and numerical features.
- Prepared the dataset for exploratory data analysis and machine learning.

---

# 3. Exploratory Data Analysis (EDA)

EDA was performed to understand the structure, distributions, patterns, and unusual observations in the dataset.

The analysis was divided into:

1. Categorical Feature Analysis
2. Numerical Feature Analysis
3. Outlier Detection and Analysis

---

## 3.1 Categorical Feature Analysis

Bar charts were created for the following categorical features:

- School
- Address
- Family Size
- Parent Status
- Mother's Job
- Father's Job
- Reason for Choosing School
- Guardian
- School Support
- Family Support
- Paid Classes
- Activities
- Nursery
- Higher Education
- Internet Access
- Romantic Relationship

Each bar chart displays the frequency of each category.

The categorical feature visualization was saved as:

`01_categ_features.png`

### Categorical Feature Analysis Code

```python
cat_columns = [
    'school', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob',
    'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
    'nursery', 'higher', 'internet', 'romantic'
]

fig, axs = plt.subplots(4, 4, figsize=(18, 10))
axes = axs.flatten()

for ax, col in zip(axes, cat_columns):
    count = df[col].value_counts()

    bars = sns.barplot(
        x=count.index,
        y=count.values,
        ax=ax
    )

    bars.bar_label(bars.containers[0])
    ax.set_title(col)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=20)

plt.tight_layout()

plt.savefig(
    "01_categ_features.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
```

---

## 3.2 Numerical Feature Analysis

The numerical features were analyzed using statistical summaries, distributions, and boxplots.

The numerical features include:

- Age
- Mother's Education (`Medu`)
- Father's Education (`Fedu`)
- Travel Time (`traveltime`)
- Study Time (`studytime`)
- Past Failures (`failures`)
- Family Relationship (`famrel`)
- Free Time (`freetime`)
- Going Out (`goout`)
- Weekday Alcohol Consumption (`Dalc`)
- Weekend Alcohol Consumption (`Walc`)
- Health (`health`)
- Absences (`absences`)
- First Period Grade (`G1`)
- Second Period Grade (`G2`)
- Result

The final grade (`G3`) is also the original source of the binary target variable and can be retained in the dataset for analysis before target construction.

---

## 3.3 Numerical Boxplots

Boxplots were created to visualize:

- Median
- Interquartile range
- Data spread
- Potential statistical outliers

The boxplots were generated using the existing `numeric_columns` list.

The visualization was saved as:

`02_numeric_features_boxplots.png`

### Boxplot Code

```python
fig, axs = plt.subplots(4, 4, figsize=(18, 10))
axes = axs.flatten()

for ax, col in zip(axes, numeric_columns):

    sns.boxplot(
        y=df[col],
        ax=ax
    )

    ax.set_title(col)
    ax.set_xlabel("")
    ax.set_ylabel("Value")

plt.tight_layout()

plt.savefig(
    "02_numeric_features_boxplots.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
```

---

# 4. Outlier Detection and Analysis

Outlier detection was performed using **boxplots** and the **Interquartile Range (IQR) method**.

The IQR is calculated as:

```text
IQR = Q3 - Q1
```

The lower and upper boundaries are:

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Values below the lower bound or above the upper bound were identified as **potential statistical outliers**.

---

## 4.1 Outlier Detection Code

```python
for col in numeric_columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower_bound) |
        (df[col] > upper_bound)
    ]

    print(f"{col}: {len(outliers)} outliers")
```

---

## 4.2 Outlier Detection Results

The IQR method identified the following potential outliers:

| Feature | Potential Outliers |
|---|---:|
| Age | 1 |
| Medu | 0 |
| Fedu | 2 |
| Traveltime | 8 |
| Studytime | 27 |
| Failures | 83 |
| Famrel | 26 |
| Freetime | 19 |
| Goout | 0 |
| Dalc | 18 |
| Walc | 0 |
| Health | 0 |
| Absences | 15 |
| G1 | 0 |
| G2 | 13 |
| Result | 0 |

---

# 5. Interpretation of Outliers

The presence of a statistical outlier does **not automatically mean that the observation is incorrect**.

Several variables in this dataset are discrete or ordinal features. For example:

- `studytime`
- `traveltime`
- `failures`
- `famrel`
- `freetime`
- `Dalc`

For these variables, a value identified as an outlier by the IQR method can still represent a valid student response.

Therefore:

> **Statistical outlier ≠ incorrect data**

Outliers were investigated before deciding whether they should be removed or transformed.

---

# 6. Absences Outlier Analysis

The `absences` feature showed the clearest presence of extreme observations in the boxplot.

The IQR method identified **15 potential outliers**.

The identified absence values ranged from:

- **21 absences**
- to **75 absences**

These observations were further investigated using:

- `absences`
- `G1`
- `G2`
- `Result`

---

## 6.1 Absence Outlier Table

| Absences | G1 | G2 | Result |
|---:|---:|---:|---:|
| 21 | 17 | 18 | 1 |
| 22 | 6 | 6 | 0 |
| 22 | 9 | 9 | 0 |
| 22 | 13 | 10 | 1 |
| 23 | 13 | 13 | 1 |
| 24 | 18 | 18 | 1 |
| 25 | 7 | 10 | 1 |
| 26 | 7 | 6 | 0 |
| 28 | 10 | 9 | 0 |
| 30 | 8 | 8 | 0 |
| 38 | 8 | 9 | 0 |
| 40 | 13 | 11 | 1 |
| 54 | 11 | 12 | 1 |
| 56 | 9 | 9 | 0 |
| 75 | 10 | 9 | 0 |

---

## 6.2 Absence Outlier Investigation

The 15 potential absence outliers were inspected together with their academic grades and classification result.

No obvious data-entry errors were identified from these records.

The high absence values can represent genuine student behavior, so they were **retained in the dataset**.

The observation with 75 absences is particularly extreme, but it was also retained because there was no evidence that it was an invalid or incorrectly entered value.

---

# 7. Outlier Treatment Decision

The project follows a cautious approach to outlier treatment.

The following process was used:

```text
Detect Outliers
      ↓
Investigate Extreme Values
      ↓
Check Whether Values Are Valid
      ↓
Identify Possible Data Errors
      ↓
Remove/Transform Only If Necessary
      ↓
Continue with Machine Learning
```

### Final Decision

- Potential statistical outliers were identified using the IQR method.
- The outliers were investigated instead of being automatically deleted.
- High absence values were retained.
- Extreme values in ordinal features were retained when they represented valid responses.
- No obvious data-entry errors were identified in the investigated absence outliers.
- No automatic outlier removal was performed.

This approach preserves genuine student observations and avoids unnecessary loss of information.

---


# 8. Feature Encoding and Standardization

After completing the exploratory data analysis and outlier investigation, the categorical and numerical features were prepared for machine learning.

The following preprocessing techniques were used:

- **One-Hot Encoding** for categorical features
- **Standardization** for numerical features

---

## 8.1 Separating Features and Target

The target variable `Result` was separated from the input features.

```python
X = df.drop(columns=['Result'])
Y = df['Result']

# 8. Key Dataset Insights

## Target Variable

The target variable contains:

- **Pass:** 265 students (67%)
- **Fail:** 130 students (33%)

The dataset is moderately imbalanced but remains suitable for binary classification.

---

## Student Demographics

The EDA showed that:

- Most students are between **15 and 18 years old**.
- Most students attend **GP School**.
- Most students live in **urban areas**.
- Most students belong to families with more than three members.

---

## Education

The analysis indicates that:

- Most parents have medium to high education levels.
- Study time is commonly concentrated around category **2**.
- The majority of students have **no previous academic failures**.
- A large proportion of students intend to pursue higher education.

---

## Family and Social Factors

The analysis indicates that:

- Many students report good family relationships.
- Family educational support is common.
- Most students are cared for by their mother.
- Students show different levels of social activity and family support.

---

## Lifestyle

The EDA indicates that:

- Weekday alcohol consumption is generally low.
- Weekend alcohol consumption is slightly higher.
- Most students report relatively good health.
- Most students have internet access at home.

---

## Academic Performance

The academic features show that:

- G1 and G2 represent the first and second period grades.
- Most students have moderate-to-high academic scores.
- G1 does not show major statistical outliers.
- G2 contains a small number of potential extreme observations.
- Absences contain several high-value observations.
- High absence values were investigated and retained as potentially valid observations.

---

# 9. Important EDA Findings

The most important findings from the exploratory analysis are:

1. The dataset contains **395 students**.
2. The target variable contains **265 passing students and 130 failing students**.
3. The target classes are moderately imbalanced.
4. Categorical features were analyzed using frequency bar charts.
5. Numerical features were analyzed using distributions and boxplots.
6. Potential outliers were identified using the IQR method.
7. `Absences` contained **15 potential outliers**.
8. The highest investigated absence value was **75**.
9. The absence outliers were examined using G1, G2, and Result.
10. No obvious data-entry errors were identified among the investigated absence observations.
11. High absence observations were therefore retained.
12. Several ordinal features contained statistical outliers, but these values can represent valid student responses.
13. Outliers were not automatically removed from the dataset.
14. The dataset is now ready for the next preprocessing and machine learning stages.

---

# 10. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

# 11. Machine Learning Pipeline

The next stage of the project is to prepare the data for machine learning.

The planned workflow is:

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Outlier Detection
   ↓
Outlier Investigation
   ↓
Categorical Encoding
   ↓
Feature Scaling
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
```

---

# 12. Machine Learning Models

The following classification algorithms will be trained and evaluated:

## Logistic Regression

A linear classification algorithm that will be used as a baseline model for binary classification.

## Naive Bayes

A probabilistic classification algorithm based on Bayes' theorem.

## K-Nearest Neighbors (KNN)

A distance-based classification algorithm that predicts the class based on nearby observations.

---

# 13. Model Evaluation

The models will be evaluated and compared using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

These metrics will be used to determine which model performs best at predicting whether a student will Pass or Fail.

---

# 14. Expected Outcome

The final machine learning system will classify whether a student is likely to:

- **Pass**
- **Fail**

based on demographic, academic, family, social, lifestyle, and other available features.

The project demonstrates how **data cleaning, exploratory data analysis, outlier investigation, preprocessing, and machine learning** can be combined to analyze and predict student academic performance.
