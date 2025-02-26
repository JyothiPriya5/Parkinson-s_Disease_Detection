import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn import metrics
import matplotlib.pyplot as plt
import pickle

# Load dataset
df = pd.read_csv('data.csv')

# Drop 'name' column and convert 'status' to uint8
df.drop(['name'], axis=1, inplace=True)
df['status'] = df['status'].astype('uint8')

# Print the value counts of 'status'
print(df['status'].value_counts())

# Split features and target
X = df.drop(['status'], axis=1)
y = df['status']

# Handle class imbalance
sm = SMOTE(random_state=300)
X, y = sm.fit_resample(X, y)

# Scale features
scaler = MinMaxScaler((-1, 1))
X_features = scaler.fit_transform(X)
Y_labels = y

# Split the dataset into training and testing sets (80 - 20)
X_train, X_test, y_train, y_test = train_test_split(X_features, Y_labels, test_size=0.20, random_state=20)

# Create and fit the Random Forest model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Make predictions
predRF = rfc.predict(X_test)
print(classification_report(y_test, predRF))

# Hyperparameter tuning with GridSearchCV
param_grid = { 
    'n_estimators': range(100, 300, 25),
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': range(1, 10),
    'criterion': ['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
CV_rfc.fit(X_train, y_train)

# Train the final model with the best parameters
best_params = CV_rfc.best_params_
rfc1 = RandomForestClassifier(random_state=200, **best_params)
rfc1.fit(X_train, y_train)

# Make predictions with the tuned model
predRFC = rfc1.predict(X_test)
print(classification_report(y_test, predRFC))

# Plot confusion matrix using ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(rfc1, X_test, y_test, cmap=plt.cm.Blues) 
plt.title('Confusion Matrix for Random Forest', y=1.1)
plt.show()

# ROC curve
y_pred_proba = rfc1.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr, tpr, label="AUC = " + str(auc))
plt.legend(loc=4)
plt.show()

# Save the model using pickle
with open('rf_clf.pkl', 'wb') as file:
    pickle.dump(rfc1, file)

print("Model saved successfully!")
