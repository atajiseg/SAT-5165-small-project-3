#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'cardio_train.csv'
df = pd.read_csv(file_path, delimiter=';')  # Use the correct delimiter

# Display the first few rows of the dataset
print(df.head())

# Step 1: Plot histograms for numerical predictors
numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
df[numerical_features].hist(bins=30, figsize=(12, 10), grid=False)
plt.suptitle('Histograms of Numerical Predictors')
plt.show()

# Step 2: Plot bar plots for categorical predictors
categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
plt.figure(figsize=(12, 8))
for i, feature in enumerate(categorical_features, 1):
    plt.subplot(3, 2, i)
    sns.countplot(x=feature, data=df)
    plt.title(f'Frequency Distribution of {feature}')
plt.tight_layout()
plt.show()

# Step 3: Create boxplots to check for outliers among numerical features
plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 2, i)
    sns.boxplot(x=df[feature])
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()

# Step 4: Generate a correlation matrix to examine relationships
correlation_matrix = df[numerical_features + ['cardio']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()

# Step 5: Preprocessing the dataset for model training
# Encode categorical features (convert categorical columns to numeric)
label_encoder = LabelEncoder()

# Encoding 'gender' as an example (and other categorical variables if needed)
df['gender'] = label_encoder.fit_transform(df['gender'])  # Male = 1, Female = 0

# Encode other categorical columns (cholesterol, gluc, smoke, alco, active)
df['cholesterol'] = label_encoder.fit_transform(df['cholesterol'])
df['gluc'] = label_encoder.fit_transform(df['gluc'])
df['smoke'] = label_encoder.fit_transform(df['smoke'])
df['alco'] = label_encoder.fit_transform(df['alco'])
df['active'] = label_encoder.fit_transform(df['active'])

# Features and target variable
X = df.drop(['cardio'], axis=1)  # Features (all columns except 'cardio')
y = df['cardio']  # Target variable (cardio disease: 0 or 1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Define the models (Logistic Regression, KNN, Decision Tree, Random Forest, SVM)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True)  # Use probability=True for AUC calculation
}

# Step 7: Train each model and calculate performance metrics
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for AUC calculation

    # Step 8: Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)  # AUC requires predicted probabilities
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)

    # Step 9: Print performance metrics
    print(f"Accuracy: {accuracy:.2f}")
    print(f"AUC (Area Under ROC): {auc:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")

    # Step 10: Confusion Matrix (for further evaluation)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Visualizing the confusion matrix (optional)
    plt.figure(figsize=(6,6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()




# In[ ]:




