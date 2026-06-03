import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
df = pd.read_csv("credit_data.csv")
df.fillna(df.mean(numeric_only=True), inplace=True)
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=['object']).columns:
    df[column] = label_encoder.fit_transform(df[column])
a = df.drop("Creditworthy", axis=1)
y = df["Creditworthy"]
a_train, a_test, y_train, y_test = train_test_split(
    a,
    y,
    test_size=0.2,
    random_state=42
)
scaler = StandardScaler()

a_train = scaler.fit_transform(a_train)
a_test = scaler.transform(a_test) 
model = LogisticRegression()
model.fit(a_train, y_train)
y_pred = model.predict(a_test)
y_prob = model.predict_proba(a_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred) 
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)   
roc_auc = roc_auc_score(y_test, y_prob)
conf_matrix = confusion_matrix(y_test, y_pred)
print("Accuracy:", accuracy)
print("Precision:", precision)

new_customer = pd.DataFrame({
    "Income": [60000],
    "Debt": [10000],
    "Age": [28]
})

new_customer_scaled = scaler.transform(new_customer)

prediction = model.predict(new_customer_scaled)

probability = model.predict_proba(new_customer_scaled)

print("\nPrediction:", prediction[0])

print("Probability of Good Credit:", probability[0][1])















