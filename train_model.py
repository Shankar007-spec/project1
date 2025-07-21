import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import joblib
# 1️⃣ Create a dummy dataset matching your app inputs
data = pd.DataFrame({
    "Age": np.random.randint(22, 60, 300),
    "Gender": np.random.choice(["Male", "Female"], 300),
    "Education Level": np.random.choice(["High School", "Bachelor", "Master", "PhD"], 300),
    "Job Title": np.random.choice(["Developer", "Data Scientist", "Manager", "Analyst", "Engineer"], 300),
    "Years of Experience": np.random.randint(0, 35, 300)
})

# Fake salary logic: purely for demo
base_salary = 250000
data["Salary"] = (
    base_salary
    + data["Age"] * 3000
    + data["Years of Experience"] * 15000
    + data["Education Level"].map({
        "High School": 0,
        "Bachelor": 50000,
        "Master": 100000,
        "PhD": 200000
    })
    + data["Job Title"].map({
        "Developer": 200000,
        "Data Scientist": 400000,
        "Manager": 500000,
        "Analyst": 150000,
        "Engineer": 250000
    })
    + np.random.normal(0, 50000, 300)
)

# 2️⃣ Preprocessing pipeline
numeric_features = ["Age", "Years of Experience"]
numeric_transformer = StandardScaler()

categorical_features = ["Gender", "Education Level", "Job Title"]
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# 3️⃣ Train the model
X = data[["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]]
y = data["Salary"]

pipeline.fit(X, y)

# 4️⃣ Save model as `salary_prediction_pipeline.pkl`
joblib.dump(pipeline, "salary_prediction_pipeline.pkl")

print("✅ Model trained and saved as 'salary_prediction_pipeline.pkl'")
