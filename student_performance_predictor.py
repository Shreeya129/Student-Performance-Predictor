# ============================================================
#   Student Performance Predictor
#   Tech: Python, Scikit-learn, Pandas, Matplotlib, Seaborn
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score
)

import warnings
warnings.filterwarnings("ignore")

# ============================================================
# STEP 1: Generate Realistic Dataset (1000 students)
# ============================================================

np.random.seed(42)
n = 1000

attendance       = np.random.randint(50, 100, n)          # % attendance
study_hours      = np.random.randint(1, 10, n)            # hours/day
prev_score       = np.random.randint(30, 100, n)          # previous exam score
assignments_done = np.random.randint(0, 10, n)            # out of 10
gender           = np.random.choice(["Male", "Female"], n)
internet_access  = np.random.choice(["Yes", "No"], n)
sleep_hours      = np.random.randint(4, 10, n)

# Target: Pass (1) or Fail (0) — based on realistic logic
score = (
    0.35 * prev_score +
    0.25 * attendance +
    0.20 * study_hours * 7 +
    0.10 * assignments_done * 5 +
    0.10 * sleep_hours * 5 +
    np.random.normal(0, 5, n)
)
result = (score >= 55).astype(int)   # 1 = Pass, 0 = Fail

df = pd.DataFrame({
    "gender"          : gender,
    "internet_access" : internet_access,
    "attendance"      : attendance,
    "study_hours"     : study_hours,
    "prev_score"      : prev_score,
    "assignments_done": assignments_done,
    "sleep_hours"     : sleep_hours,
    "result"          : result
})

print("=" * 55)
print("        STUDENT PERFORMANCE PREDICTOR")
print("=" * 55)
print(f"\nDataset Shape  : {df.shape}")
print(f"Pass Students  : {result.sum()} ({result.mean()*100:.1f}%)")
print(f"Fail Students  : {n - result.sum()} ({(1-result.mean())*100:.1f}%)")
print("\nSample Data:")
print(df.head())

# ============================================================
# STEP 2: Preprocessing
# ============================================================

# Encode categorical columns
le = LabelEncoder()
df["gender"]          = le.fit_transform(df["gender"])           # Male=1, Female=0
df["internet_access"] = le.fit_transform(df["internet_access"])  # Yes=1, No=0

# Features & Target
X = df.drop("result", axis=1)
y = df["result"]

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print("\n✔ Preprocessing complete")
print(f"  Training samples : {len(X_train)}")
print(f"  Testing  samples : {len(X_test)}")

# ============================================================
# STEP 3: Train 3 Models
# ============================================================

models = {
    "Logistic Regression" : LogisticRegression(random_state=42),
    "Decision Tree"       : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"       : RandomForestClassifier(
                                n_estimators=100,
                                max_depth=6,
                                random_state=42
                            ),
}

results = {}

print("\n" + "=" * 55)
print("  MODEL COMPARISON")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds    = model.predict(X_test)
    acc      = accuracy_score(y_test, preds) * 100
    f1       = f1_score(y_test, preds) * 100
    results[name] = {"model": model, "preds": preds, "acc": acc, "f1": f1}
    print(f"\n  {name}")
    print(f"    Accuracy : {acc:.2f}%")
    print(f"    F1 Score : {f1:.2f}%")

# Best model
best_name  = max(results, key=lambda k: results[k]["acc"])
best_model = results[best_name]["model"]
best_preds = results[best_name]["preds"]

print(f"\n✔ Best Model : {best_name} ({results[best_name]['acc']:.2f}%)")

# ============================================================
# STEP 4: Detailed Report for Best Model
# ============================================================

print("\n" + "=" * 55)
print(f"  CLASSIFICATION REPORT — {best_name}")
print("=" * 55)
print(classification_report(y_test, best_preds, target_names=["Fail", "Pass"]))

# ============================================================
# STEP 5: Visualizations
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Student Performance Predictor — Results", fontsize=14, fontweight="bold")

# --- Plot 1: Model Accuracy Comparison ---
names = list(results.keys())
accs  = [results[n]["acc"] for n in names]
bars  = axes[0].bar(names, accs, color=["#4C72B0", "#DD8452", "#55A868"], edgecolor="black")
axes[0].set_title("Model Accuracy Comparison", fontweight="bold")
axes[0].set_ylabel("Accuracy (%)")
axes[0].set_ylim(60, 100)
for bar, acc in zip(bars, accs):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{acc:.1f}%", ha="center", va="bottom", fontweight="bold")
axes[0].set_xticklabels(names, rotation=10, ha="right")

# --- Plot 2: Confusion Matrix ---
cm = confusion_matrix(y_test, best_preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
axes[1].set_title(f"Confusion Matrix\n({best_name})", fontweight="bold")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

# --- Plot 3: Feature Importance (Random Forest) ---
rf_model = results["Random Forest"]["model"]
feat_imp  = pd.Series(rf_model.feature_importances_,
                       index=["Gender", "Internet", "Attendance",
                              "Study Hrs", "Prev Score", "Assignments", "Sleep"])
feat_imp.sort_values().plot(kind="barh", ax=axes[2], color="#4C72B0", edgecolor="black")
axes[2].set_title("Feature Importance\n(Random Forest)", fontweight="bold")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig("student_performance_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✔ Chart saved as 'student_performance_results.png'")

# ============================================================
# STEP 6: Predict a New Student (Demo)
# ============================================================

print("\n" + "=" * 55)
print("  PREDICT A NEW STUDENT")
print("=" * 55)

# Example: Female, has internet, 80% attendance, 5 hrs study,
#          prev score 70, 8 assignments done, 7 hrs sleep
new_student = pd.DataFrame([{
    "gender"          : 0,    # Female = 0
    "internet_access" : 1,    # Yes = 1
    "attendance"      : 80,
    "study_hours"     : 5,
    "prev_score"      : 70,
    "assignments_done": 8,
    "sleep_hours"     : 7
}])

new_student_scaled = scaler.transform(new_student)
prediction         = best_model.predict(new_student_scaled)[0]
probability        = best_model.predict_proba(new_student_scaled)[0]

print(f"\n  Input  : Attendance=80%, Study=5hrs, Prev Score=70, Assignments=8/10")
print(f"  Result : {'✅ PASS' if prediction == 1 else '❌ FAIL'}")
print(f"  Pass Probability : {probability[1]*100:.1f}%")
print(f"  Fail Probability : {probability[0]*100:.1f}%")

print("\n" + "=" * 55)
print("  PROJECT COMPLETE!")
print("=" * 55)
