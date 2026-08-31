
# EX NO. 3(b): DATA PREPROCESSING

## PAGE 1 — TITLE, AIM AND OBJECTIVES

### DATA PREPROCESSING USING PYTHON AND SCIKIT-LEARN

### AIM

To perform data preprocessing on a dataset using Python and Scikit-learn by handling missing values, encoding categorical data, splitting the dataset into training and testing sets, and applying feature scaling.

### OBJECTIVES

The main objectives of this experiment are:

1. To understand the importance of data preprocessing in machine learning.
2. To load and inspect a dataset using Pandas.
3. To identify independent and dependent variables.
4. To identify and handle missing values.
5. To convert categorical data into numerical form.
6. To perform Label Encoding on categorical variables.
7. To apply One-Hot Encoding to categorical data.
8. To encode the target/dependent variable.
9. To divide the dataset into training and testing datasets.
10. To apply feature scaling using StandardScaler.
11. To prepare clean and standardized data for machine learning algorithms.

### INTRODUCTION

Data preprocessing is one of the most important steps in a machine learning workflow. Real-world datasets are often incomplete, inconsistent, and contain different types of data. Machine learning algorithms generally require data to be in a suitable numerical format.

Data preprocessing transforms raw data into a clean and usable form. It can involve handling missing values, converting categorical variables into numerical values, selecting relevant features, splitting data, and scaling numerical features.

In this experiment, Python libraries such as **Pandas, NumPy, and Scikit-learn** are used to preprocess a sample dataset.

---

# PAGE 2 — THEORY OF DATA PREPROCESSING

## THEORY

### What is Data Preprocessing?

Data preprocessing is the process of preparing raw data before it is given to a machine learning algorithm.

A typical dataset may contain:

* Numerical values
* Categorical values
* Missing values
* Different scales
* Irrelevant information
* Inconsistent formats

These issues can affect the performance of machine learning models.

### Importance of Data Preprocessing

Data preprocessing is necessary because machine learning algorithms work better when the input data is properly prepared.

For example, consider a dataset containing:

| Country | Age | Salary | Purchased |
| ------- | --: | -----: | --------- |
| France  |  44 |  72000 | No        |
| Spain   |  27 |  48000 | Yes       |
| Germany |  30 |  54000 | No        |

The **Country** column contains text, whereas Age and Salary contain numerical values.

Most machine learning algorithms cannot directly process values such as:

* France
* Spain
* Germany

Therefore, categorical data must be converted into numerical representations.

### Main Steps in Data Preprocessing

The major preprocessing steps performed in this experiment are:

1. Loading the dataset
2. Understanding the dataset
3. Separating X and Y
4. Handling missing values
5. Encoding categorical variables
6. Encoding the target variable
7. Splitting the dataset
8. Feature scaling

### Raw Data → Preprocessed Data

```text
Raw Dataset
     ↓
Inspect Dataset
     ↓
Separate X and Y
     ↓
Handle Missing Values
     ↓
Encode Categorical Data
     ↓
Encode Target Variable
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
Preprocessed Dataset
```

---

# PAGE 3 — DATASET AND PYTHON LIBRARIES

## DATASET

The dataset used in this experiment is **Data.csv**.

The dataset contains information related to individuals and their purchasing decisions.

The major columns are:

### 1. Country

This is a categorical variable that represents the country of the individual.

Example:

```text
France
Spain
Germany
```

### 2. Age

Age represents the age of the individual and is a numerical variable.

Example:

```text
44
27
30
```

### 3. Salary

Salary represents the individual's salary and is a numerical variable.

Example:

```text
72000
48000
54000
```

### 4. Purchased

Purchased is the dependent variable or target variable.

It indicates whether the individual purchased a product.

Possible values are:

```text
Yes
No
```

## PYTHON LIBRARIES USED

### Pandas

Pandas is used for handling datasets and performing data analysis.

```python
import pandas as pd
```

It provides the DataFrame structure, which makes it easy to work with rows and columns.

### NumPy

NumPy is used for numerical operations and array manipulation.

```python
import numpy as np
```

In this experiment, `np.nan` is used to identify missing values.

### Scikit-learn

Scikit-learn provides several machine learning preprocessing tools.

The following modules are used:

* `SimpleImputer`
* `LabelEncoder`
* `OneHotEncoder`
* `train_test_split`
* `StandardScaler`

---

# PAGE 4 — DATA LOADING AND DATA INSPECTION

## PROCEDURE

### Step 1: Import Libraries and Load Dataset

Google Drive is mounted so that the dataset stored in Google Drive can be accessed.

```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np

df = pd.read_csv('/content/drive/MyDrive/Datasets/Data.csv')
```

The `read_csv()` function reads the CSV file and stores it as a Pandas DataFrame.

### Display First Few Records

```python
df.head()
```

The `head()` function displays the first five rows of the dataset by default.

This helps us understand:

* Column names
* Data values
* Data types
* General structure of the dataset

### Step 2: Check Dataset Information

```python
df.info()
print(df.shape)
```

The `df.info()` function provides information about:

* Number of rows
* Number of columns
* Column names
* Data types
* Non-null values

The `df.shape` attribute returns:

```text
(number of rows, number of columns)
```

### Why Dataset Inspection is Important

Before preprocessing, it is important to understand the dataset.

Inspection helps identify:

* Missing values
* Numerical columns
* Categorical columns
* Target variables
* Number of observations

This prevents errors during later preprocessing operations.

---

# PAGE 5 — SEPARATING VARIABLES AND HANDLING MISSING VALUES

## Step 3: Separate Independent and Dependent Variables

The independent variables are stored in `x`, while the dependent variable is stored in `y`.

```python
x = df[['Country', 'Age', 'Salary']]
y = df[['Purchased']].values
```

Here:

### Independent Variables — X

```text
Country
Age
Salary
```

These variables are used to predict the target.

### Dependent Variable — Y

```text
Purchased
```

The dependent variable represents the outcome that the machine learning model attempts to predict.

### Convert X into an Array

```python
x = df[['Country', 'Age', 'Salary']].values
```

The `.values` property converts the DataFrame into a NumPy array.

---

## Step 4: Handling Missing Values

Real-world datasets may contain missing values.

For example:

```text
France    44    72000
Spain     27    48000
Germany   NaN   54000
```

The missing Age value must be handled before applying machine learning algorithms.

### SimpleImputer

Scikit-learn provides the `SimpleImputer` class for replacing missing values.

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='mean')
```

The strategy used here is:

```text
mean
```

This means that a missing numerical value is replaced with the mean of the available values in that column.

### Applying the Imputer

```python
imputer.fit(x[:, 1:3])

x[:, 1:3] = imputer.transform(x[:, 1:3])
```

Here:

```python
x[:, 1:3]
```

selects the numerical columns:

```text
Age
Salary
```

The imputer calculates the mean and replaces missing numerical values.

---

# PAGE 6 — ENCODING CATEGORICAL DATA

## Step 5: Label Encoding

Machine learning algorithms generally require numerical input.

The Country column contains categorical values.

For example:

```text
France
Spain
Germany
```

Label Encoding converts categories into numerical labels.

```python
from sklearn.preprocessing import LabelEncoder

label_encoder_x = LabelEncoder()

x[:, 0] = label_encoder_x.fit_transform(x[:, 0])

print(x)
```

The encoder assigns numerical labels to the countries.

For example, the output may resemble:

```text
France  → 0
Germany → 1
Spain   → 2
```

The exact numerical assignment depends on the encoder's ordering.

### Advantages of Label Encoding

* Simple to implement
* Converts text categories into numbers
* Useful for certain categorical variables
* Requires little additional memory

### Limitation

For nominal categories, numerical labels can sometimes incorrectly suggest an order between categories.

For example:

```text
France = 0
Germany = 1
Spain = 2
```

This does **not** mean Spain is greater than Germany or Germany is greater than France.

---

## Step 6: One-Hot Encoding

One-Hot Encoding is another method for representing categorical data numerically.

```python
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()

x_country = onehotencoder.fit_transform(
    df.Country.values.reshape(-1, 1)
).toarray()

print(x_country)
```

Instead of assigning one number to each country, One-Hot Encoding creates separate columns.

For example:

| France | Germany | Spain |
| -----: | ------: | ----: |
|      1 |       0 |     0 |
|      0 |       1 |     0 |
|      0 |       0 |     1 |

Only one category receives the value `1` for each observation.

### Difference

**Label Encoding**

```text
France → 0
Germany → 1
Spain → 2
```

**One-Hot Encoding**

```text
France  → [1,0,0]
Germany → [0,1,0]
Spain   → [0,0,1]
```

One-Hot Encoding is commonly used when categories do not have a meaningful order.

---

# PAGE 7 — TARGET ENCODING, TRAIN-TEST SPLIT AND SCALING

## Encoding the Dependent Variable

The `Purchased` column contains categorical values such as:

```text
Yes
No
```

These values are converted into numerical values using LabelEncoder.

```python
labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

print(y)
```

The output becomes numerical, for example:

```text
No  → 0
Yes → 1
```

This makes the target variable suitable for machine learning algorithms.

---

## Step 7: Splitting the Dataset

The dataset is divided into:

* Training dataset
* Testing dataset

The training dataset is used to train the machine learning model.

The testing dataset is used to evaluate the model.

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=0
)
```

### Test Size

```python
test_size=0.2
```

means that approximately:

```text
80% → Training data
20% → Testing data
```

### Random State

```python
random_state=0
```

ensures that the same split can be reproduced when the program is executed again.

---

## Step 8: Feature Scaling

Different numerical features may have very different ranges.

For example:

```text
Age    → 20–60
Salary → 30,000–100,000
```

Salary has a much larger numerical range than Age.

This can affect algorithms that depend on distances or numerical magnitude.

Therefore, feature scaling is performed.

### StandardScaler

```python
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
```

StandardScaler standardizes the features using their mean and standard deviation.

The standardization formula is:

$$
z = \frac{x-\mu}{\sigma}
$$

Where:

* `x` = original value
* `μ` = mean
* `σ` = standard deviation
* `z` = standardized value

After scaling, the features are generally centered around zero with a standard deviation close to one.

---

# PAGE 8 — COMPLETE PROGRAM, RESULT AND CONCLUSION

## COMPLETE PROGRAM

```python
# Step 1: Import libraries and load dataset
from google.colab import drive

drive.mount('/content/drive')

import pandas as pd
import numpy as np

df = pd.read_csv('/content/drive/MyDrive/Datasets/Data.csv')

# Display dataset
df.head()

# Step 2: Check dataset information
df.info()
print(df.shape)

# Step 3: Separate independent and dependent variables
x = df[['Country', 'Age', 'Salary']]
y = df[['Purchased']].values

# Convert X into array
x = df[['Country', 'Age', 'Salary']].values

# Step 4: Handle missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    missing_values=np.nan,
    strategy='mean'
)

imputer.fit(x[:, 1:3])

x[:, 1:3] = imputer.transform(x[:, 1:3])

print(x)

# Step 5: Encode categorical data
from sklearn.preprocessing import LabelEncoder

label_encoder_x = LabelEncoder()

x[:, 0] = label_encoder_x.fit_transform(x[:, 0])

print(x)

# Step 6: One-Hot Encoding
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder()

x_country = onehotencoder.fit_transform(
    df.Country.values.reshape(-1, 1)
).toarray()

print(x_country)

# Encode dependent variable
labelencoder_y = LabelEncoder()

y = labelencoder_y.fit_transform(y)

print(y)

# Step 7: Split dataset into training and testing sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=0
)

print(x_train)
print(x_test)
print(y_train)

# Step 8: Feature Scaling
from sklearn.preprocessing import StandardScaler

sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

print(x_train)
print(x_test)
```

## RESULT

The given dataset was successfully preprocessed using Python and Scikit-learn.

The following preprocessing operations were performed successfully:

* Dataset loading using Pandas
* Dataset inspection using `head()`, `info()`, and `shape`
* Separation of independent and dependent variables
* Conversion of data into NumPy arrays
* Handling of missing numerical values using `SimpleImputer`
* Label Encoding of categorical data
* One-Hot Encoding of the Country column
* Label Encoding of the Purchased column
* Splitting the dataset into training and testing sets
* Feature scaling using `StandardScaler`

## ADVANTAGES OF DATA PREPROCESSING

1. Improves the quality of input data.
2. Handles missing values.
3. Converts categorical data into numerical form.
4. Reduces problems caused by different feature scales.
5. Makes datasets suitable for machine learning algorithms.
6. Can improve model training and performance.
7. Helps produce more consistent results.

## APPLICATIONS

Data preprocessing is widely used in:

* Healthcare data analysis
* Banking and finance
* Customer purchase prediction
* Fraud detection
* Recommendation systems
* Image processing
* Natural language processing
* Predictive analytics
* Business intelligence
* Classification and regression problems

## CONCLUSION

Thus, the given dataset was successfully preprocessed using Python and Scikit-learn. Missing numerical values were handled using `SimpleImputer`, categorical data was converted into numerical form using LabelEncoder and One-Hot Encoding, and the dependent variable was encoded using LabelEncoder. The dataset was then divided into training and testing sets using `train_test_split`. Finally, feature scaling was performed using `StandardScaler`.

Therefore, the dataset was successfully transformed into a suitable format for applying machine learning algorithms.
