# Artificial Neural Networks (ANN) — Classification & Regression

Two beginner-friendly deep learning projects built with **Python and PyTorch**. These Jupyter notebooks demonstrate how to prepare tabular data, build and train an Artificial Neural Network (ANN), and evaluate its predictions for two different types of machine learning problems.

## Projects at a Glance

| Notebook | Problem | Dataset | What the model predicts |
|---|---|---|---|
| `ANN_classification.ipynb` | Multiclass classification | Date fruit dataset | One of seven date fruit classes |
| `ANN_regression.ipynb` | Regression | Power plant dataset | Continuous power output (`PE`) |

**Classification** predicts a category, such as a type of fruit. **Regression** predicts a number, such as the amount of electricity generated.

## Technologies

- Python
- PyTorch — neural networks, training, and tensor operations
- Pandas and NumPy — data handling
- scikit-learn — splitting, preprocessing, and evaluation
- Matplotlib — plotting regression training and validation losses
- Jupyter Notebook — running the projects step by step

## Project Structure

```text
ANN-Projects/
├── ANN_classification.ipynb
├── ANN_regression.ipynb
├── date_fruit_dataset.csv       # Required for classification; supply separately
├── powerplant _dataset.csv      # Required for regression; supply separately
├── bestmodel.pt                 # Generated when regression training runs
└── README.md
```

> **Important:** The uploaded notebooks do not include the CSV datasets. Add your own copies of the corresponding datasets. The regression notebook currently expects the exact filename `powerplant _dataset.csv` (including the space before `_`). You can alternatively change its `pd.read_csv(...)` path to match your actual file.

## Getting Started

### 1. Install the requirements

Use Python 3 and install the libraries:

```bash
pip install numpy pandas torch scikit-learn matplotlib notebook
```

### 2. Prepare the datasets

Place `date_fruit_dataset.csv` and `powerplant _dataset.csv` in the working directory from which the notebooks will read them. Check that their column names match those used in the notebooks.

### 3. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open either notebook and run its cells in order. You can also open the notebooks in VS Code or another compatible Jupyter environment.

---

## Project 1: ANN for Date Fruit Classification

### Goal

Build a neural network that uses numerical features of date fruits to predict which of **seven classes** each sample belongs to.

### How it works

1. **Load the data:** Read `date_fruit_dataset.csv` with Pandas.
2. **Separate features and labels:** Use every column except `Class` as input features (`X`); use `Class` as the target (`y`).
3. **Encode labels:** Convert class names into numerical labels using `LabelEncoder`.
4. **Split the dataset:** Use 80% for training and 20% for testing (`random_state=42`).
5. **Scale features:** Fit `StandardScaler` on the training inputs and transform both training and testing inputs.
6. **Create tensors:** Convert inputs to `float32` tensors and class labels to `long` tensors.
7. **Create data loaders:** Feed the model batches of 32 samples; shuffle training batches.
8. **Train the ANN:** Run forward propagation, calculate classification loss, perform backpropagation, and update weights.
9. **Evaluate:** Compare predicted class indices with true labels to calculate test accuracy.

### Neural Network Architecture

```text
Input Layer (number of dataset features)
       ↓
Linear Layer (64 neurons)
       ↓
ReLU Activation
       ↓
Linear Layer (64 neurons)
       ↓
ReLU Activation
       ↓
Output Layer (7 class scores)
```

| Setting | Value |
|---|---|
| Framework | PyTorch |
| Hidden layers | 2 |
| Neurons per hidden layer | 64 |
| Activation | ReLU |
| Output neurons | 7 |
| Loss function | CrossEntropyLoss |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 32 |
| Epochs | 100 |
| Train/test split | 80% / 20% |

**Why CrossEntropyLoss?** It is designed for classification problems where the model chooses among multiple classes. The model outputs raw class scores; `CrossEntropyLoss` handles the required log-softmax internally.

### Evaluation

The notebook prints the number of test samples, the number classified correctly, and accuracy:

```text
Accuracy (%) = (Correct predictions / Total test samples) × 100
```

The actual accuracy depends on running the notebook with the dataset. No measured accuracy is claimed here.

**Note:** The final notebook cell mentions PCA (dimensionality reduction), but PCA has **not** been implemented in the supplied code.

---

## Project 2: ANN for Power Plant Regression

### Goal

Build a neural network that predicts a power plant's **electrical power output (`PE`)** from environmental measurements.

### Dataset Features

| Column | Meaning | Role |
|---|---|---|
| `AT` | Temperature | Input |
| `V` | Vacuum | Input |
| `AP` | Atmospheric pressure | Input |
| `RH` | Relative humidity | Input |
| `PE` | Power output | Prediction target |

### How it works

1. **Load the data:** Read `powerplant _dataset.csv`.
2. **Check missing values:** Use `df.isnull().sum()`.
3. **Select inputs and target:** Predict `PE` from the other columns.
4. **Split the data:** 80% training and 20% testing (`random_state=42`).
5. **Standardize inputs:** Fit `StandardScaler` on training features and apply it to both splits. The target `PE` is not scaled.
6. **Create tensors and batches:** Use `float32` tensors and a batch size of 32. Reshape target tensors to have one value per row.
7. **Train the ANN:** Use mean squared error and the Adam optimizer over 100 epochs.
8. **Track losses:** Record training and validation losses after each epoch and plot them.
9. **Save the best model:** When validation loss improves, save the weights as `bestmodel.pt`.
10. **Evaluate:** Reload the saved model, calculate training/testing MSE and test R², and display actual vs. predicted values.

### Neural Network Architecture

```text
Input Layer (4 features)
       ↓
Linear Layer (6 neurons)
       ↓
ReLU Activation
       ↓
Linear Layer (6 neurons)
       ↓
ReLU Activation
       ↓
Output Layer (1 numerical prediction)
```

| Setting | Value |
|---|---|
| Framework | PyTorch |
| Input features | 4 |
| Hidden layers | 2 |
| Neurons per hidden layer | 6 |
| Activation | ReLU |
| Output neurons | 1 |
| Loss function | MSELoss |
| Optimizer | Adam (default settings) |
| Batch size | 32 |
| Epochs | 100 |
| Train/test split | 80% / 20% |
| Saved model | `bestmodel.pt` |

### Evaluation Metrics

**Mean Squared Error (MSE)** measures the average squared difference between predictions and actual values. **Lower is better.**

```text
MSE = mean((Actual - Predicted)²)
```

**R² score** measures how well predictions explain variation in the target. A value of 1 indicates perfect predictions; values can be negative for a poor model.

The notebook also creates a training/validation loss graph and a table comparing predicted and actual power output. Exact numerical results are not specified because they depend on executing the notebook.

**Important evaluation note:** The regression notebook uses the same 20% split both for validation during training/model selection and for final testing. For a more reliable assessment on unseen data, create a separate validation set and reserve the test set for final evaluation only.

---

## Classification vs. Regression

| Aspect | Classification ANN | Regression ANN |
|---|---|---|
| Objective | Predict a class | Predict a number |
| Example | Date fruit variety | Power plant output |
| Output neurons | 7 | 1 |
| Loss function | Cross-entropy | Mean squared error |
| Evaluation | Accuracy | MSE and R² |
| Saved best model | Not implemented | Implemented |

## Key Deep Learning Concepts

- **Artificial Neural Network (ANN):** A model made of connected layers of neurons that learns patterns from examples.
- **Forward propagation:** Passing input data through the network to generate predictions.
- **Loss function:** A number measuring how different the predictions are from the expected answers.
- **Backpropagation:** Calculating gradients to determine how model weights should change.
- **Optimizer:** The algorithm that updates the weights to reduce loss.
- **Epoch:** One full pass through the training dataset.
- **Batch:** A smaller group of training examples processed together.
- **ReLU:** An activation function that helps the network learn nonlinear patterns.
- **Feature scaling:** Adjusting input feature scales so training is more stable.

## Possible Improvements

- Create a separate training/validation/test split for both projects.
- Add a confusion matrix and per-class precision, recall, and F1-score to classification.
- Experiment with learning rates, hidden-layer sizes, and epoch counts.
- Add early stopping and reproducible random seeds.
- Save preprocessing objects alongside trained models for later inference.
- Implement PCA only if experiments show it is useful for the classification dataset.

## Conclusion

These notebooks demonstrate two core applications of Artificial Neural Networks with PyTorch: **multiclass classification** and **numerical regression**. Together, they provide a practical introduction to data preprocessing, model construction, training loops, evaluation, and model saving.

---

*Created as learning projects for practicing deep learning with Python and PyTorch.*
