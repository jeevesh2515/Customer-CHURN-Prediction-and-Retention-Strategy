import pandas as pd     
import lime
from lime import lime_tabular
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Assuming X_train is defined somewhere above
X_train = ...  # Define or load X_train here
X = pd.DataFrame(X_train)  # Define X as a DataFrame

# Preprocess X_train if necessary
scaler = StandardScaler()
X_train_res = scaler.fit_transform(X_train)

# Initialize LIME explainer
explainer = lime_tabular.LimeTabularExplainer(X_train_res, 
                                              feature_names=X.columns, 
                                              class_names=['No Churn', 'Churn'], 
                                              discretize_continuous=True)

# Assuming X_test is defined somewhere above
X_test = ...  # Define or load X_test here

scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

# Load or define the random forest model
rf_model = joblib.load('rf_model.pkl')

# Choose a sample customer from the test set
i = 25  # Index of the customer to explain
exp = explainer.explain_instance(X_test_scaled[i], rf_model.predict_proba, num_features=6)

# Show the explanation
exp.show_in_notebook(show_table=True)

# Predict churn probabilities
y_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]  # Assuming binary classification
y_pred_rf = rf_model.predict(X_test_scaled)  # Add this line to define y_pred_rf

# Add the churn probabilities and predictions to a DataFrame for analysis
results_df = pd.DataFrame(X_test_scaled, columns=X.columns)
results_df['Churn_Probability'] = y_proba_rf
results_df['Predicted_Churn'] = y_pred_rf

# Show high-risk customers (churn probability > 0.6)
high_risk_customers = results_df[results_df['Churn_Probability'] > 0.6]
print(high_risk_customers.head())

# Group customers by churn risk level
high_risk_customers = results_df[results_df['Churn_Probability'] >= 0.7]
medium_risk_customers = results_df[(results_df['Churn_Probability'] < 0.7) & (results_df['Churn_Probability'] >= 0.4)]
low_risk_customers = results_df[results_df['Churn_Probability'] < 0.4]

# Visualize churn probability distribution
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(results_df['Churn_Probability'], bins=10, kde=True)
plt.title('Churn Probability Distribution')
plt.xlabel('Churn Probability')
plt.ylabel('Number of Customers')
plt.show()



data = pd.read_csv('/Users/jeeveshsingale/PycharmProjects/pythonProject1/customer_churn.csv')  