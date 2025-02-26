import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib  # Import joblib for saving the scaler
import matplotlib.pyplot as plt

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

# Save the MinMaxScaler to a file
joblib.dump(scaler, 'minmax_scaler.pkl')

# Continue with your model training, e.g., train-test split and model fitting
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.20, random_state=20)

