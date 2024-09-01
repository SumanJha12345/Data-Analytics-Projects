import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
train_data = pd.read_csv('train_data.csv')
print(train_data.info())
summary_stats_train = train_data.describe()
print("Summary Statistics for Training Data:\n", summary_stats_train)
# Histograms for numerical features in the training dataset
train_data.hist(figsize=(15, 15), bins=20)
plt.suptitle('Histograms of Numerical Features (Training Data)', y=0.92)
plt.show()
# Box plots for numerical features in the training dataset
num_features_train = train_data.select_dtypes(include=['int64', 'float64']).columns
num_plots = len(num_features_train)
num_rows = (num_plots // 3) + (num_plots % 3 > 0)  # Calculate the number of rows needed

plt.figure(figsize=(15, 5 * num_rows))
for i, feature in enumerate(num_features_train, 1):
    plt.subplot(num_rows, 3, i)
    sns.boxplot(x='Is high risk', y=feature, data=train_data)
    plt.title(f'Box Plot of {feature} by Credit Card Approval')

plt.tight_layout()
plt.show()
# Scatter plots for numerical features in the training dataset
sns.pairplot(train_data, hue='Is high risk')
plt.suptitle('Pair Plot of Numerical Features (Training Data)', y=1.02)
plt.show()
# Define numerical and categorical features
num_features_train = train_data.select_dtypes(include=['int64', 'float64']).columns
cat_features_train = train_data.select_dtypes(include=['object']).columns
# Plotting numerical features
num_features_train = train_data.select_dtypes(include=['int64', 'float64']).columns
num_plots = len(num_features_train)
num_rows = (num_plots // 3) + (num_plots % 3 > 0)  # Calculate the number of rows needed

plt.figure(figsize=(15, 5 * num_rows))
for i, feature in enumerate(num_features_train, 1):
    plt.subplot(num_rows, 3, i)
    if train_data[feature].nunique() > 10:
        sns.histplot(train_data[feature], bins=30, kde=True)
    else:
        sns.countplot(x=feature, data=train_data)
    plt.title(f'Distribution of {feature}')
# Plotting categorical features
plt.figure(figsize=(15, 10))
for i, feature in enumerate(cat_features_train, 1):
    plt.subplot(3, 3, i)
    sns.countplot(x=feature, data=train_data, order=train_data[feature].value_counts().index)
    plt.title(f'Distribution of {feature}')
    plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.show()
# Scatter plots for pairs of numerical features
plt.figure(figsize=(15, 10))
sns.pairplot(train_data, hue='Is high risk', diag_kind='kde')
plt.suptitle('Pair Plot of Numerical Features by Credit Card Approval (Training Data)', y=1.02)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to leave space for the suptitle
plt.show()
# Box plots for the relationship between numerical features and the target variable
plt.figure(figsize=(15, 10))
for i, feature in enumerate(num_features_train, 1):
    plt.subplot(num_rows, 3, i)
    sns.boxplot(x='Is high risk', y=feature, data=train_data)
    plt.title(f'Box Plot of {feature} by Credit Card Approval')

plt.tight_layout(pad=2)  # Increase the padding between subplots
plt.show()
# Box plots for the relationship between categorical features and the target variable
plt.figure(figsize=(15, 15))
for i, feature in enumerate(cat_features_train, 1):
    plt.subplot(len(cat_features_train) // 3 + 1, 3, i)
    sns.countplot(x=feature, hue='Is high risk', data=train_data)
    plt.title(f'Count Plot of {feature} by Credit Card Approval')
    plt.xticks(rotation=45, ha='right')

plt.tight_layout(pad=2)  # Increase the padding between subplots
plt.show()


# Verify that missing values have been handled
print("Remaining Missing Values:\n", train_data.isnull().sum())

from sklearn.preprocessing import LabelEncoder
# Binning Numerical Features
bins = [0, 25, 35, 50, 100]
labels = ['Young', 'Adult', 'Middle-aged', 'Senior']
train_data['Age_Group'] = pd.cut(train_data['Age'], bins=bins, labels=labels)

# Creating Interaction Terms
train_data['Income_Family'] = train_data['Income'] * train_data['Family member count']
# Label Encoding Categorical Features
label_encoder = LabelEncoder()
categorical_features = ['Gender', 'Has a car', 'Has a property', 'Employment status', 'Education level', 'Marital status', 'Dwelling', 'Job title']
for feature in categorical_features:
    train_data[feature+'_Encoded'] = label_encoder.fit_transform(train_data[feature])
# Display the updated dataset with new features
print("Updated Dataset with Engineered Features:\n", train_data.head())

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Separate features and target variable
X = train_data.drop(columns=['Is high risk'])
y = train_data['Is high risk']

# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Define preprocessing steps
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean
    ('scaler', MinMaxScaler())  # Scale numerical features to a specific range
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with the most frequent value
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
])

# Bundle preprocessing for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# Logistic Regression with increased max_iter
logistic_model = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', LogisticRegression(max_iter=1000))])  # Increase max_iter

# Fit the model with preprocessed data
logistic_model.fit(X_train, y_train)

# Perform predictions
predictions = logistic_model.predict(X_valid)

# Decision Tree
dt_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', DecisionTreeClassifier(random_state=42))])
dt_model.fit(X_train, y_train)

# Random Forest
rf_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', RandomForestClassifier(random_state=42))])
rf_model.fit(X_train, y_train)

# Gradient Boosting
gb_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('model', GradientBoostingClassifier(random_state=42))])
gb_model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Function for model evaluation
def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    confusion_mat = confusion_matrix(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, model.predict_proba(X)[:, 1])

    return accuracy, precision, recall, f1, confusion_mat, roc_auc

# Evaluate Logistic Regression
logistic_metrics = evaluate_model(logistic_model, X_valid, y_valid)

# Evaluate Decision Tree
dt_metrics = evaluate_model(dt_model, X_valid, y_valid)

# Evaluate Random Forest
rf_metrics = evaluate_model(rf_model, X_valid, y_valid)

# Evaluate Gradient Boosting
gb_metrics = evaluate_model(gb_model, X_valid, y_valid)

# Print the evaluation metrics for each model
print("Logistic Regression Metrics:", logistic_metrics)
print("Decision Tree Metrics:", dt_metrics)
print("Random Forest Metrics:", rf_metrics)
print("Gradient Boosting Metrics:", gb_metrics)

from sklearn.metrics import accuracy_score
# Assuming you have the true labels and predicted labels for each model

# Calculate accuracy for each model
accuracy_logistic = accuracy_score(y_true_logistic, y_pred_logistic)
accuracy_dt = accuracy_score(y_true_dt, y_pred_dt)
accuracy_rf = accuracy_score(y_true_rf, y_pred_rf)
accuracy_gb = accuracy_score(y_true_gb, y_pred_gb)

# Print accuracy for each model
print("Logistic Regression Accuracy:", accuracy_logistic)
print("Decision Tree Accuracy:", accuracy_dt)
print("Random Forest Accuracy:", accuracy_rf)
print("Gradient Boosting Accuracy:", accuracy_gb)

test_data = pd.read_csv('test_data.csv')
# Add missing columns and fill missing values (if any)
missing_columns = set(train_data.columns) - set(test_data.columns)
for column in missing_columns:
    test_data[column] = 0  # Fill with zeros or use appropriate values based on your preprocessing
# Ensure the columns are in the same order
test_data = test_data[train_data.columns]

#true labels are in the 'Is high risk' column of the test_data DataFrame
true_labels = test_data['Is high risk']
# Logistic Regression
logistic_accuracy_test = accuracy_score(true_labels, logistic_predictions_test)
print(f'Logistic Regression Accuracy on Test Data: {logistic_accuracy_test}')

# Decision Tree
dt_accuracy_test = accuracy_score(true_labels, dt_predictions_test)
print(f'Decision Tree Accuracy on Test Data: {dt_accuracy_test}')

# Random Forest
rf_accuracy_test = accuracy_score(true_labels, rf_predictions_test)
print(f'Random Forest Accuracy on Test Data: {rf_accuracy_test}')

# Gradient Boosting
gb_accuracy_test = accuracy_score(true_labels, gb_predictions_test)
print(f'Gradient Boosting Accuracy on Test Data: {gb_accuracy_test}')

import joblib
# Assuming 'logistic_model' is the best-performing model
best_model = logistic_model

# Save the best model to a file
joblib.dump(best_model, 'best_model.pkl')
!pip install joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Load training data
train_data = pd.read_csv('train_data.csv')

# Define features and target variable
X = train_data.drop('Is high risk', axis=1)
y = train_data['Is high risk']

# Define numeric and categorical features
numeric_features = ['Income', 'Age', 'Employment length', 'Family member count', 'Account age']
categorical_features = ['Gender', 'Has a car', 'Has a property', 'Children count', 'Employment status',
                        'Education level', 'Marital status', 'Dwelling', 'Has a mobile phone',
                        'Has a work phone', 'Has a phone', 'Has an email', 'Job title']

# Create transformers for numeric and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers into a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create and train the pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', model)])

# Fit the pipeline to the training data
pipeline.fit(X, y)
# Save the model to a file
joblib.dump(pipeline, 'best_model.joblib')