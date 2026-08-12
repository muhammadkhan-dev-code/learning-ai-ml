# Student Math Performance Analysis using Machine Learning

## Project Overview

This project analyzes the **Student Math Performance Dataset** to understand the factors affecting students' academic performance. The dataset was preprocessed and explored using **Exploratory Data Analysis (EDA)** before applying classification algorithms.

The objective is to predict whether a student will **Pass** or **Fail** based on demographic, family, social, and academic features.

---

## Problem Type

**Binary Classification**

The original target variable (`G3`) contains the final mathematics grade (0–20). It was converted into a binary target variable:

- **Pass (1):** G3 ≥ 10
- **Fail (0):** G3 < 10

The new target variable is named **Result**.

---

## Work Completed

### 1. Data Loading
- Loaded the Student Math Performance dataset into a Pandas DataFrame.
- Inspected the dataset using:
  - `head()`
  - `tail()`
  - `shape`
  - `info()`
  - `describe()`

---

### 2. Data Cleaning

The following preprocessing steps were performed:

- Removed the unnecessary `Unnamed: 0` index column.
- Checked for missing values.
- Checked for duplicate records.
- Verified data types.
- Created the binary target variable (`Result`).
- Prepared the dataset for classification.

---

### 3. Exploratory Data Analysis (EDA)

EDA was performed to understand the dataset before building machine learning models.

#### Categorical Features

Bar charts were created for:

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

Each bar chart displays the frequency of every category.

---

#### Numerical Features

Histograms were plotted for:

- Age
- Mother's Education
- Father's Education
- Travel Time
- Study Time
- Past Failures
- Family Relationship
- Free Time
- Going Out
- Weekday Alcohol Consumption
- Weekend Alcohol Consumption
- Health
- Absences
- First Period Grade (G1)
- Second Period Grade (G2)
- Result

These plots help visualize the distribution of each numerical variable.

---
# 4. Outlier Detection and Analysis

Outlier detection was performed on the numerical features using **boxplots** and the **Interquartile Range (IQR) method**.

The IQR was calculated using:

```text
IQR = Q3 - Q1
```
The IQR method identified 15 potential outliers, with absence values ranging from 21 to 75.



## Key Insights

### Dataset

- Total Students: **395**
- Total Features: **33**
- Classification Target: **Result**

---

### Target Variable

- Pass: **265 students (67%)**
- Fail: **130 students (33%)**

The dataset is **moderately imbalanced** but still suitable for binary classification.

---

### Student Demographics

- Most students are between **15 and 18 years old**.
- Most students attend **GP School**.
- Most students live in **urban areas**.
- Most students belong to families with more than three members.

---

### Education

- Most parents have medium to high education levels.
- Most students study approximately **2 hours** per week (study time category 2).
- The majority of students have **no previous academic failures**.

---

### Family & Social Factors

- Most students report good family relationships.
- Most students receive family educational support.
- Most students are cared for by their mother.
- Most students intend to pursue higher education.

---

### Lifestyle

- Weekday alcohol consumption is generally very low.
- Weekend alcohol consumption is slightly higher.
- Most students report good health.
- Most students have internet access at home.

---

### Academic Performance

- First-period (G1) and second-period (G2) grades are approximately normally distributed.
- Most students have relatively few absences.
- A small number of students have very high absence counts, indicating possible outliers.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (for upcoming machine learning models)

---

## Machine Learning Pipeline

The next steps in the project are:

- Encode categorical variables.
- Scale numerical features.
- Split the dataset into training and testing sets.
- Train and evaluate:
  - Logistic Regression
  - Naive Bayes
  - K-Nearest Neighbors (KNN)
- Compare model performance using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix

---

## Expected Outcome

The final model will classify whether a student is likely to **Pass** or **Fail** based on the available demographic, academic, family, and lifestyle features.