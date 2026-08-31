# Iris Species Classification

## 1. Project Overview

This project uses the Iris dataset to explore and build a machine learning classification model.

The goal is to predict the species of an Iris flower using its sepal and petal measurements.

### Target Variable

`Species`

The dataset contains three classes:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

---

## 2. Dataset Overview

The dataset contains 150 rows and 6 columns.

| Column          | Description                 |
| --------------- | --------------------------- |
| `Id`            | Unique identifier           |
| `SepalLengthCm` | Sepal length in centimeters |
| `SepalWidthCm`  | Sepal width in centimeters  |
| `PetalLengthCm` | Petal length in centimeters |
| `PetalWidthCm`  | Petal width in centimeters  |
| `Species`       | Target variable             |

The `Id` column was removed because it does not provide useful information for classification.

---

## 3. Data Quality Check

The dataset was checked for common data quality problems.

- No missing values
- No null values
- No duplicate rows

Therefore, the dataset is clean and ready for analysis and machine learning.

---

## 4. Target Variable Analysis

The target variable is `Species`.

The class distribution is:

| Species         | Count |
| --------------- | ----: |
| Iris-setosa     |    50 |
| Iris-versicolor |    50 |
| Iris-virginica  |    50 |

---

## 5. Exploratory Data Analysis

Several EDA techniques were performed to understand the dataset.

### Histogram Analysis

The main observation was that `PetalLengthCm` and `PetalWidthCm` show clear groups of values, suggesting that these features may be useful for distinguishing between species.

### Correlation Analysis

A correlation heatmap was used to understand relationships between numerical features.

The strongest positive correlation was found between:

- `PetalLengthCm`
- `PetalWidthCm`

This indicates that flowers with longer petals generally tend to have wider petals.

### Boxplot Analysis

Boxplots were used to compare the feature distributions across the three species.

The petal features showed clearer differences between species compared with the sepal features.

### Pairplot Analysis

The pairplot showed that the three species form relatively distinct groups.

The clearest separation was observed when comparing:

- `PetalLengthCm`
- `PetalWidthCm`

This suggests that petal measurements are particularly useful for species classification.

---

## 6. Data Preparation

The data was divided into:

- Features (`X`)
- Target (`y`)

The `Species` column was used as the target variable, while the four measurement columns were used as features.

The dataset was split into:

- 80% Training Data
- 20% Testing Data

Stratified splitting was used to maintain the balanced class distribution.

---

## 7. Machine Learning Models

Three classification models were tested:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Gaussian Naive Bayes

The models were first trained using the original features without scaling.

After that, `StandardScaler` was applied and the same models were trained again.

---

## 8. Model Performance

The accuracy results before and after feature scaling are:

| Model               | Before Scaling | After Scaling |
| ------------------- | -------------: | ------------: |
| Logistic Regression |         96.67% |        93.33% |
| KNN                 |        100.00% |        93.33% |
| Naive Bayes         |         96.67% |        96.67% |

### Observations

- Logistic Regression decreased from **96.67% to 93.33%** after scaling.
- KNN decreased from **100% to 93.33%** after scaling.
- Naive Bayes remained unchanged at **96.67%**.
- The best result in this experiment was **KNN before scaling with 100% accuracy**.

---

## 9. Confusion Matrix

Confusion matrices were used to understand the correct and incorrect predictions for each model.
The confusion matrix compares:

- Actual Species
- Predicted Species

Correct predictions appear on the diagonal, while incorrect predictions appear outside the diagonal.

---
