# Iris Species Classification — Exploratory Data Analysis

## 1. Project Overview

This project uses the **Iris dataset** to explore and eventually build a machine learning classification model that predicts the species of an Iris flower based on its physical measurements.

The target variable is:

* `Species`

The three classes are:

* `Iris-setosa`
* `Iris-versicolor`
* `Iris-virginica`

---

## 2. Dataset Overview

The dataset contains **150 observations** and the following columns:

| Column          | Description                                   |
| --------------- | --------------------------------------------- |
| `Id`            | Unique identifier for each observation        |
| `SepalLengthCm` | Sepal length in centimeters                   |
| `SepalWidthCm`  | Sepal width in centimeters                    |
| `PetalLengthCm` | Petal length in centimeters                   |
| `PetalWidthCm`  | Petal width in centimeters                    |
| `Species`       | Target variable representing the Iris species |

### Dataset Shape

```text
Rows: 150
Columns: 6
```

---

## 3. Data Quality Check

The dataset was checked for common data quality problems.

### Missing Values

There are **no missing values** in the dataset.

```text
Missing values: 0
```

### Null Values

There are **no null values** in the dataset.

### Duplicate Values

No duplicate rows were found in the dataset.

Therefore, the dataset is clean enough to continue with the exploratory data analysis and machine learning pipeline.

---

## 4. Target Variable Analysis

The target variable is `Species`.

Using:

```python
y.value_counts()
```

the class distribution is:

```text
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
```

Each class contains exactly **50 observations**.

### Conclusion

The target variable is **perfectly balanced**.

This is useful for classification because there is no significant class imbalance that would cause one species to dominate the model's learning process.

---

## 5. Feature Analysis

The dataset contains four numerical features:

```text
SepalLengthCm
SepalWidthCm
PetalLengthCm
PetalWidthCm
```

The `Id` column does not represent a meaningful biological measurement, so it should not be used as a machine learning feature.

It can be removed using:

```python
df = df.drop('Id', axis=1)
```

---

## 6. Histogram Analysis

Histograms were created to understand the distribution of the numerical features.

### SepalLengthCm

`SepalLengthCm` values are approximately distributed between **4.3 and 7.9 cm**.

The values are spread across several ranges, with higher frequencies around the middle of the distribution. No obvious extreme outliers are visible from the histogram.

### SepalWidthCm

`SepalWidthCm` values range approximately from **2.0 to 4.4 cm**.

Most observations are concentrated around **2.5–3.5 cm**, with the highest frequency around approximately 3 cm.

### PetalLengthCm

`PetalLengthCm` shows a more interesting distribution.

There is a large concentration of observations around **1–2 cm**, followed by another group at larger values.

This indicates that the feature may contain strong information for distinguishing between the different Iris species.

### PetalWidthCm

`PetalWidthCm` also shows clearly separated groups of values.

The observations form groups around smaller, medium, and larger petal widths.

This suggests that `PetalWidthCm` is likely to be a strong predictive feature for classifying the Iris species.

---

## 7. Important EDA Observation

The distributions of `PetalLengthCm` and `PetalWidthCm` appear to have multiple groups.

This is likely because the dataset contains three different species with different physical characteristics.

Therefore, the petal-related features may provide strong separation between:

```text
Iris-setosa
Iris-versicolor
Iris-virginica
```

This is an important observation because the goal of the project is to classify the species.

---

## 8. Current EDA Status

The following analysis has been completed:

* [x] Load the dataset
* [x] Understand dataset shape
* [x] Check column names
* [x] Check data types
* [x] Check missing values
* [x] Check null values
* [x] Check duplicate rows
* [x] Analyze target variable
* [x] Check class distribution
* [x] Analyze numerical feature distributions using histograms
* [x] Identify potentially useful features

---

## 9. Next Steps

The next stage of the analysis will focus on understanding the relationship between the features and the target variable.

Planned steps:

1. **Boxplots by Species**

   * Compare each feature across the three Iris species.
   * Identify differences and possible outliers.

2. **Correlation Analysis**

   * Understand relationships between numerical features.

3. **Pairplot**

   * Visualize relationships between all features while separating observations by species.

4. **Feature and Target Separation**

   ```python
   X = df.drop('Species', axis=1)
   y = df['Species']
   ```

5. **Train/Test Split**

   * Split the dataset into training and testing data.

6. **Feature Scaling**

   * Apply scaling where required by the selected machine learning algorithms.

7. **Model Training**

   * Train multiple classification algorithms.

8. **Model Evaluation**

   * Compare models using accuracy, precision, recall, F1-score, and confusion matrix.

---
---
* PetalLengthCm and PetalWidthCm have the strongest positive correlation, while SepalWidthCm has relatively weaker relationships with the other features. The strong correlation between petal length and petal width indicates that these features carry related information and may be highly useful for Iris species classification
---

## Conclusion

So far, the Iris dataset has been successfully inspected and cleaned. It contains **150 observations with three equally represented species and no missing, null, or duplicate values**.

The histogram analysis shows that the **petal measurements, particularly `PetalLengthCm` and `PetalWidthCm`, appear to provide strong separation between different species**.

The next important step is to perform **boxplot analysis and pairplot visualization by species** before moving into the machine learning modeling stage.
