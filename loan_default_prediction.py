# Loan Default Prediction - Extracted from Jupyter Notebook
# -------------------------------------------------------
# This script contains all the code extracted from the Jupyter Notebook
# with added comments for better readability and understanding.



# -----------------------------
# Code Cell 1
# -----------------------------
#  1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings
warnings.filterwarnings('ignore')



# -----------------------------
# Code Cell 2
# -----------------------------
#  Load the dataset
df = pd.read_csv("hmeq.csv")

#  Display basic structure
print("Initial shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
df.describe()



# -----------------------------
# Code Cell 3
# -----------------------------
df.describe(include='all')


# -----------------------------
# Code Cell 4
# -----------------------------
# Fill missing values for EDA
df['REASON'].fillna(df['REASON'].mode()[0], inplace=True)
df['JOB'].fillna(df['JOB'].mode()[0], inplace=True)
df['YOJ'].fillna(df['YOJ'].median(), inplace=True)
df['LOAN'].fillna(df['LOAN'].median(), inplace=True)
df['MORTDUE'].fillna(df['MORTDUE'].median(), inplace=True)
df['VALUE'].fillna(df['VALUE'].median(), inplace=True)

# Q1: Range of LOAN
loan_min = df['LOAN'].min()
loan_max = df['LOAN'].max()
print(f"Q1 - Loan Amount Range: Min = {loan_min}, Max = {loan_max}")
print()
print()

# Q2: Distribution of YOJ
yoj_stats = df['YOJ'].describe()
print("Q2 - YOJ Statistics:")
print(yoj_stats)
print()
print()


# Q3: Unique categories in REASON
reason_unique_count = df['REASON'].nunique()
print(f"Q3 - Number of unique categories in REASON: {reason_unique_count}")
print()
print()


# Q4: Most common category in JOB
job_most_common = df['JOB'].mode()[0]
print(f"Q4 - Most common JOB category: {job_most_common}")
print()
print()


# Q5: Default rate by REASON
default_rate_by_reason = df.groupby('REASON')['BAD'].mean()
print("Q5 - Default rate by REASON:")
print(default_rate_by_reason)
print()
print()


# Q6: Loan amount by default status
loan_by_default = df.groupby('BAD')['LOAN'].describe()
print("Q6 - Loan amount by BAD (default status):")
print(loan_by_default)
print()
print()


# Q7: Property value by default status
value_by_default = df.groupby('BAD')['VALUE'].describe()
print("Q7 - Property value by BAD (default status):")
print(value_by_default)
print()
print()


# Q8: Mortgage due by default status
mortdue_by_default = df.groupby('BAD')['MORTDUE'].describe()
print("Q8 - Mortgage due by BAD (default status):")
print(mortdue_by_default)
print()
print()


# Visualize YOJ distribution
sns.histplot(df['YOJ'], bins=30, kde=True)
plt.title("Distribution of Years at Job (YOJ)")
plt.xlabel("YOJ")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Visualize Loan amount by default
sns.boxplot(x='BAD', y='LOAN', data=df)
plt.title("Loan Amount by Default Status")
plt.xlabel("BAD (0 = No Default, 1 = Default)")
plt.ylabel("Loan Amount")
plt.grid(True)
plt.show()

# Visualize Property Value by default
sns.boxplot(x='BAD', y='VALUE', data=df)
plt.title("Property Value by Default Status")
plt.xlabel("BAD")
plt.ylabel("Property Value")
plt.grid(True)
plt.show()

# Visualize Mortgage Due by default
sns.boxplot(x='BAD', y='MORTDUE', data=df)
plt.title("Mortgage Due by Default Status")
plt.xlabel("BAD")
plt.ylabel("Mortgage Due")
plt.grid(True)
plt.show()



# -----------------------------
# Code Cell 5
# -----------------------------
numerical_columns = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ', 'CLAGE', 'NINQ', 'CLNO', 'DEBTINC']
categorical_columns = ['REASON', 'JOB', 'BAD']

#  Display summary statistics for numerical columns
num_summary = df[numerical_columns].describe()
print("Summary Statistics for Numerical Columns:\n")
print(num_summary)

#  Display value counts for categorical columns
for col in categorical_columns:
    print(f"\nValue Counts for Categorical Feature: {col}")
    print(df[col].value_counts())


# -----------------------------
# Code Cell 6
# -----------------------------
numerical_bivariate = df.groupby('BAD')[numerical_columns].mean()
print("Mean of Numerical Features grouped by Loan Default (BAD):\n")
print(numerical_bivariate)

# Proportion of BAD=1 in each category for categorical variables
for col in categorical_columns:
    print(f"\nProportion of Default by {col}:")
    print(pd.crosstab(df[col], df['BAD'], normalize='index'))


# -----------------------------
# Code Cell 7
# -----------------------------
numerical_features = ['LOAN', 'MORTDUE', 'VALUE', 'YOJ', 'DEROG', 'DELINQ',
                      'CLAGE', 'NINQ', 'CLNO', 'DEBTINC', 'BAD']
correlation_matrix = df[numerical_features].corr()
print("Correlation Matrix (Rounded):\n")
print(correlation_matrix.round(2))
# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix of Numerical Features", fontsize=16)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()


# -----------------------------
# Code Cell 8
# -----------------------------
def treat_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    before_outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()

    # Cap outliers
    df[column] = df[column].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))
    after_outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()

    print(f"{column} - Outliers before: {before_outliers}, after treatment: {after_outliers}")

# Apply outlier treatment on all numerical columns
for col in numerical_features:
    treat_outliers_iqr(df, col)


# -----------------------------
# Code Cell 9
# -----------------------------
#Check missing values before imputation
missing_before = df.isnull().sum()

# Impute missing values

# Categorical features: fill with mode (most frequent value)
df['REASON'].fillna(df['REASON'].mode()[0], inplace=True)  # Reason for loan
df['JOB'].fillna(df['JOB'].mode()[0], inplace=True)        # Job category

# Numerical features: fill with median to handle skewness
df['YOJ'].fillna(df['YOJ'].median(), inplace=True)          # Years on job
df['LOAN'].fillna(df['LOAN'].median(), inplace=True)        # Loan amount
df['MORTDUE'].fillna(df['MORTDUE'].median(), inplace=True)  # Mortgage due
df['VALUE'].fillna(df['VALUE'].median(), inplace=True)      # Property value
df['CLAGE'].fillna(df['CLAGE'].median(), inplace=True)      # Age of oldest credit line
df['CLNO'].fillna(df['CLNO'].median(), inplace=True)        # Number of credit lines
df['DEBTINC'].fillna(df['DEBTINC'].median(), inplace=True)  # Debt-to-income ratio

# Features that represent counts: fill with zero (logical default)
df['DEROG'].fillna(0, inplace=True)     # Major derogatory reports
df['DELINQ'].fillna(0, inplace=True)    # Delinquent credit lines
df['NINQ'].fillna(0, inplace=True)      # Number of recent inquiries

# Check missing values after imputation
missing_after = df.isnull().sum()

# Create a summary DataFrame showing changes
missing_summary = pd.DataFrame({
    'Missing Before': missing_before,
    'Missing After': missing_after
})

# Filter only features that had missing values
missing_summary = missing_summary[missing_summary['Missing Before'] > 0]

# Display the summary
print("Missing Value Treatment Summary:\n")
print(missing_summary)



# -----------------------------
# Code Cell 10
# -----------------------------

# Compute statistical summaries to extract insights
loan_range = (df['LOAN'].min(), df['LOAN'].max())
yoj_distribution = df['YOJ'].describe()
reason_counts = df['REASON'].value_counts()
job_counts = df['JOB'].value_counts()

# Default rate by REASON
reason_default_rate = df.groupby('REASON')['BAD'].mean()

# Average loan amount for defaulters vs non-defaulters
avg_loan_by_bad = df.groupby('BAD')['LOAN'].mean()

# Correlation between VALUE and BAD
correlation_value_bad = df[['VALUE', 'BAD']].corr().iloc[0, 1]

# Average MORTDUE for defaulters vs non-defaulters
avg_mortdue_by_bad = df.groupby('BAD')['MORTDUE'].mean()

# Display computed values
{
    "Loan Range": loan_range,
    "YOJ Summary": yoj_distribution,
    "REASON Counts": reason_counts.to_dict(),
    "JOB Counts": job_counts.to_dict(),
    "Default Rate by REASON": reason_default_rate.to_dict(),
    "Avg LOAN by BAD": avg_loan_by_bad.to_dict(),
    "Correlation (VALUE vs BAD)": correlation_value_bad,
    "Avg MORTDUE by BAD": avg_mortdue_by_bad.to_dict()
}



# -----------------------------
# Code Cell 11
# -----------------------------

df = pd.read_csv("hmeq.csv")
# Fill missing categorical features with mode
df['REASON'].fillna(df['REASON'].mode()[0], inplace=True)
df['JOB'].fillna(df['JOB'].mode()[0], inplace=True)

# Fill missing numerical features with median or zero as appropriate
df['YOJ'].fillna(df['YOJ'].median(), inplace=True)
df['LOAN'].fillna(df['LOAN'].median(), inplace=True)
df['MORTDUE'].fillna(df['MORTDUE'].median(), inplace=True)
df['VALUE'].fillna(df['VALUE'].median(), inplace=True)
df['DEROG'].fillna(0, inplace=True)
df['DELINQ'].fillna(0, inplace=True)
df['CLAGE'].fillna(df['CLAGE'].median(), inplace=True)
df['NINQ'].fillna(0, inplace=True)
df['CLNO'].fillna(df['CLNO'].median(), inplace=True)
df['DEBTINC'].fillna(df['DEBTINC'].median(), inplace=True)

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df, columns=['REASON', 'JOB'], drop_first=True)

from sklearn.model_selection import train_test_split

# Ensure correct y (with both classes)
y = df_encoded['BAD'] .astype(int)
X = df_encoded.drop('BAD', axis=1)

# Redo the train-test split using stratify=y
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Confirm class distribution after splitting
print("y_train value counts:\n", y_train.value_counts())
print("y_test value counts:\n", y_test.value_counts())


log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train, y_train)

# Predict on test data
y_pred = log_model.predict(X_test)

#  Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

#  Print results
print("Accuracy Score:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)



# -----------------------------
# Code Cell 12
# -----------------------------

# Initialize the Decision Tree classifier with default parameters
tree_model = DecisionTreeClassifier(random_state=42)

# Fit the model on training data
tree_model.fit(X_train, y_train)

# Make predictions on test data
y_pred_tree = tree_model.predict(X_test)

# Evaluate the model
accuracy_tree = accuracy_score(y_test, y_pred_tree)
conf_matrix_tree = confusion_matrix(y_test, y_pred_tree)
class_report_tree = classification_report(y_test, y_pred_tree)

# Print the results
print("Accuracy Score:", accuracy_tree)
print("\nConfusion Matrix:\n", conf_matrix_tree)
print("\nClassification Report:\n", class_report_tree)



# -----------------------------
# Code Cell 13
# -----------------------------

df = pd.read_csv("hmeq.csv")



# Encode categorical columns
df_encoded = pd.get_dummies(df, columns=['REASON', 'JOB'], drop_first=True)

# Feature and target split
X = df_encoded.drop('BAD', axis=1)
y = df_encoded['BAD'].astype(int)

#  Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Define hyperparameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 10, None],
    'min_samples_leaf': [1, 5, 10, 20]
}

# Initialize model and GridSearchCV
tree = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=tree,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Evaluate the best model
best_tree = grid_search.best_estimator_
y_pred_best_tree = best_tree.predict(X_test)

accuracy_best_tree = accuracy_score(y_test, y_pred_best_tree)
conf_matrix_best_tree = confusion_matrix(y_test, y_pred_best_tree)
class_report_best_tree = classification_report(y_test, y_pred_best_tree)
best_params = grid_search.best_params_

# Print results
print("Best Hyperparameters:", best_params)
print("Accuracy:", accuracy_best_tree)
print("\nConfusion Matrix:\n", conf_matrix_best_tree)
print("\nClassification Report:\n", class_report_best_tree)



# -----------------------------
# Code Cell 14
# -----------------------------

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=100,         # number of trees
    max_depth=None,           # let trees expand fully
    min_samples_leaf=1,       # minimum samples in a leaf
    random_state=42,
    class_weight='balanced'   # handle class imbalance
)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
class_report_rf = classification_report(y_test, y_pred_rf)

#  Print evaluation results
print("Accuracy Score:", accuracy_rf)
print("\nConfusion Matrix:\n", conf_matrix_rf)
print("\nClassification Report:\n", class_report_rf)



# -----------------------------
# Code Cell 15
# -----------------------------
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from scipy.stats import randint

# Define the parameter distribution
param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Initialize RandomizedSearchCV
random_search_rf = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=10,               # Number of parameter combinations to try
    scoring='f1',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

# Fit the model
random_search_rf.fit(X_train, y_train)

# Evaluate the best model
best_rf_random = random_search_rf.best_estimator_
y_pred_best_rf_random = best_rf_random.predict(X_test)

accuracy_rf_random = accuracy_score(y_test, y_pred_best_rf_random)
conf_matrix_rf_random = confusion_matrix(y_test, y_pred_best_rf_random)
class_report_rf_random = classification_report(y_test, y_pred_best_rf_random)
best_rf_random_params = random_search_rf.best_params_

# Print the results
print("Best Hyperparameters:", best_rf_random_params)
print("Accuracy:", accuracy_rf_random)
print("\nConfusion Matrix:\n", conf_matrix_rf_random)
print("\nClassification Report:\n", class_report_rf_random)



# -----------------------------
# Code Cell 16
# -----------------------------
# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#  Define the parameter grid for tuning
param_grid = {
    'n_estimators': [100, 200],            # number of trees
    'max_depth': [5, 10, None],            # depth of each tree
    'min_samples_split': [2, 5, 10],       # min samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],         # min samples required at a leaf node
    'criterion': ['gini', 'entropy']       # function to measure quality of split
}

# Initialize Random Forest model
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Initialize GridSearchCV
grid_search_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

# Fit the model to training data
grid_search_rf.fit(X_train, y_train)

# Get the best estimator
best_rf = grid_search_rf.best_estimator_

#  Predict and evaluate on test data
y_pred_best_rf = best_rf.predict(X_test)
accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)
conf_matrix_best_rf = confusion_matrix(y_test, y_pred_best_rf)
class_report_best_rf = classification_report(y_test, y_pred_best_rf)
best_rf_params = grid_search_rf.best_params_

# Print results
print("Best Hyperparameters:", best_rf_params)
print("Accuracy:", accuracy_best_rf)
print("\nConfusion Matrix:\n", conf_matrix_best_rf)
print("\nClassification Report:\n", class_report_best_rf)

