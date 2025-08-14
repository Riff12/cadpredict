import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import ADASYN
import joblib

# Load dataset
df = pd.read_csv("CAD.csv", sep=';')  # Gunakan delimiter ';'

# Periksa kolom
print("Kolom dalam dataset:", df.columns.tolist())

# Pastikan kolom 'Cath' ada
if 'Cath' not in df.columns:
    raise ValueError("Kolom 'Cath' tidak ditemukan. Periksa dataset atau nama kolom.")

# Pisahkan fitur dan target
X = df.drop(columns='Cath')
y = LabelEncoder().fit_transform(df['Cath'])  # 0: Normal, 1: Cad

# Identifikasi kolom numerik dan kategorikal
cat_cols = X.select_dtypes(include='object').columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Pipeline Preprocessing
numeric_pipeline = Pipeline([('scaler', StandardScaler())])
categorical_pipeline = Pipeline([('encoder', OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore'))])
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols)
])

# Transformasi data
X_processed = preprocessor.fit_transform(X)

# Cek NaN
if np.isnan(X_processed).any():
    print("⚠️ Masih ada NaN setelah preprocessing.")
    exit()

# Oversampling dengan ADASYN
X_resampled, y_resampled = ADASYN(random_state=42).fit_resample(X_processed, y)

# Base learners & meta learner
rf = RandomForestClassifier(random_state=42)
svm = SVC(probability=True, random_state=42)
nb = GaussianNB()
logreg = LogisticRegression(max_iter=1000)
stack_model = StackingClassifier(
    estimators=[('rf', rf), ('svm', svm), ('nb', nb)],
    final_estimator=logreg,
    cv=5
)

# Evaluasi dengan Cross-Validation
cv_results = cross_validate(
    stack_model,
    X_resampled,
    y_resampled,
    cv=10,
    scoring=['accuracy', 'precision', 'recall', 'f1'],
    return_train_score=False
)

# Output hasil evaluasi
print("\n=== EVALUASI CROSS-VALIDATION (10-FOLD) SETELAH ADASYN ===")
print(f"CV Accuracy   : {np.mean(cv_results['test_accuracy']):.4f}")
print(f"CV Precision  : {np.mean(cv_results['test_precision']):.4f}")
print(f"CV Recall     : {np.mean(cv_results['test_recall']):.4f}")
print(f"CV F1-Score   : {np.mean(cv_results['test_f1']):.4f}")

# Train model pada seluruh data dan simpan
stack_model.fit(X_resampled, y_resampled)
joblib.dump(stack_model, 'stack_model.joblib')
joblib.dump(preprocessor, 'preprocessor.joblib')
print("Model dan preprocessor berhasil disimpan!")