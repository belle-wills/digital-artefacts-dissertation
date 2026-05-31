import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load dataset
df = pd.read_csv("fitness_coaching_dataset.csv")

print("\nLabel counts before encoding:")
print(df["coaching_label"].value_counts())

# encode text columns
encoders = {}

categorical_cols = [
    "user_level",
    "exercise",
    "cycle_phase",
    "fatigue_level",
    "performance_trend",
    "coaching_label"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# features and target
X = df.drop("coaching_label", axis=1)
y = df["coaching_label"]

# split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# scale numeric values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# train model
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=1000,
    random_state=42,
    learning_rate_init=0.001,
    verbose=True
)

model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")

# readable class names
label_names = encoders["coaching_label"].classes_

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=range(len(label_names)),
    target_names=label_names,
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# save training loss graph
plt.figure()
plt.plot(model.loss_curve_)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("training_loss.png")
#plt.show()

# save model, encoders and scaler
joblib.dump(model, "ai_coach_model.pkl")
joblib.dump(encoders, "label_encoders.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nSaved:")
print("- ai_coach_model.pkl")
print("- label_encoders.pkl")
print("- scaler.pkl")
print("- training_loss.png")