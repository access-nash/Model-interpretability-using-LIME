import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import graphviz
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'


# Load data
train_df = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Model Interpretability/Model_Interpretability_Dataset/fraudTrain.csv')
train_df['trans_date_trans_time']=pd.to_datetime(train_df['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S', errors = 'coerce')
train_df = train_df.drop(columns=['Unnamed: 0', 'trans_num', 'unix_time', 'merch_lat', 'merch_long','zip', 'lat', 'long'])
train_df["dob"] = pd.to_datetime(train_df["dob"], format='%Y-%m-%d')
train_df["age"] = ((pd.to_datetime('today') - train_df["dob"])// pd.Timedelta(days=365.2425)).astype(int)

train_df.columns
train_df.dtypes
missing_values = train_df.isnull().sum()
print(missing_values)

test_df = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Model Interpretability/Model_Interpretability_Dataset/fraudTest.csv')
test_df['trans_date_trans_time']=pd.to_datetime(test_df['trans_date_trans_time'], format='%Y-%m-%d %H:%M:%S', errors = 'coerce')
test_df = test_df.drop(columns=['Unnamed: 0', 'trans_num', 'unix_time', 'merch_lat', 'merch_long','zip', 'lat', 'long'])
test_df["dob"] = pd.to_datetime(test_df["dob"], format='%Y-%m-%d')
test_df["age"] = ((pd.to_datetime('today') - test_df["dob"])// pd.Timedelta(days=365.2425)).astype(int)

missing_values = test_df.isnull().sum()
print(missing_values)

target_col = 'is_fraud'

# Preprocess data
#categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
#print(f"Categorical columns: {categorical_cols}")
relevant_features = ['category', 'amt', 'gender', 'city', 'state', 'job', 'age']


categorical_cols = ['category', 'gender', 'city', 'state', 'job']

label_encoders = {}
for col in categorical_cols:
    all_values = pd.concat([train_df[col], test_df[col]]).astype(str)
    le = LabelEncoder()
    le.fit(all_values)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))
    label_encoders[col] = le


X_train = train_df[relevant_features].copy()
y_train = train_df[target_col]
X_test = test_df[relevant_features].copy()
y_test = test_df[target_col]

X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)

#Interpretable Decision tree model
dt = DecisionTreeClassifier(max_depth=5, random_state=10)
dt.fit(X_train, y_train)

# Accuracy scores
train_acc = accuracy_score(y_train, dt.predict(X_train))
test_acc = accuracy_score(y_test, dt.predict(X_test))
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Visualize decision tree
dot_data = export_graphviz(dt, out_file=None,
                           feature_names=X_train.columns,
                           class_names=['Not Fraud', 'Fraud'],
                           filled=True, rounded=True,
                           special_characters=True, max_depth=3)

graph = graphviz.Source(dot_data)
graph.render("fraud_tree", format="png", cleanup=True)


# Random Forest Model (Black Box)


rf = RandomForestClassifier(n_estimators=200, max_depth=5,
                            min_samples_leaf=100, n_jobs=-1,
                            random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# Accuracy scores
rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))
print(f"Training Accuracy: {rf_train_acc:.4f}")
print(f"Test Accuracy: {rf_test_acc:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'variable': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\nTop Features by Importance:")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['variable'], feature_importance['importance'])
plt.xlabel('Importance')
plt.title('Feature Importance for Fraud Detection')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()


# GLOBAL SURROGATE MODEL


# Get black box predictions (probabilities)
new_target = rf.predict_proba(X_train)[:, 1]  # Probability of fraud

# Train surrogate decision tree on black box predictions
dt_surrogate = DecisionTreeRegressor(max_depth=3, random_state=10)
dt_surrogate.fit(X_train, new_target)

# Evaluate surrogate model
surrogate_train_mse = mean_squared_error(new_target, dt_surrogate.predict(X_train))
surrogate_test_mse = mean_squared_error(rf.predict_proba(X_test)[:, 1], dt_surrogate.predict(X_test))

print(f"Surrogate Train MSE: {surrogate_train_mse:.4f}")
print(f"Surrogate Test MSE: {surrogate_test_mse:.4f}")
print(f"Surrogate R² Score (train): {dt_surrogate.score(X_train, new_target):.4f}")
print(f"Surrogate R² Score (test): {dt_surrogate.score(X_test, rf.predict_proba(X_test)[:, 1]):.4f}")

# Visualize surrogate tree
dot_data_surrogate = export_graphviz(dt_surrogate, out_file=None,
                                     feature_names=X_train.columns,
                                     class_names=['Not Fraud', 'Fraud'],
                                     filled=True, rounded=True,
                                     special_characters=True, max_depth=2)

graph_surrogate = graphviz.Source(dot_data_surrogate)
graph_surrogate.render("fraud_surrogate_tree", format="png", cleanup=True)

sample_indices = [0, 1, 2]
print("\nSample predictions comparison:")
print("Index | Black Box Prob | Surrogate Pred | Difference")

for idx in sample_indices:
    if idx < len(X_test):
        black_box_prob = rf.predict_proba(X_test.iloc[idx:idx+1])[0, 1]
        surrogate_pred = dt_surrogate.predict(X_test.iloc[idx:idx+1])[0]
        diff = abs(black_box_prob - surrogate_pred)
        print(f"{idx:5d} | {black_box_prob:.4f}        | {surrogate_pred:.4f}          | {diff:.4f}")

#  Lime interpretability


# Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    mode="classification",
    feature_names=X_train.columns.tolist(),
    class_names=['Not Fraud', 'Fraud'],
    random_state=42
)

# Analyze multiple cases
fraud_indices = y_test[y_test == 1].index[:2]
non_fraud_indices = y_test[y_test == 0].index[:2]

for i, idx in enumerate(fraud_indices):
    explanation = explainer.explain_instance(
        X_test.iloc[idx].values,
        rf.predict_proba,
        num_features=len(relevant_features)
    )

    print(f"\nFraud Case {i} - Top features:")
    for feature, weight in explanation.as_list()[:3]:
        print(f"  {feature}: {weight:.4f}")

for i, idx in enumerate(non_fraud_indices):
    explanation = explainer.explain_instance(
        X_test.iloc[idx].values,
        rf.predict_proba,
        num_features=len(relevant_features)
    )

    print(f"\nNon-Fraud Case {i} - Top features:")
    for feature, weight in explanation.as_list()[:3]:
        print(f"  {feature}: {weight:.4f}")









