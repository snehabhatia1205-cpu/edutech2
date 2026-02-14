import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="EdTech Engagement Analyzer", layout="wide")

st.title("📊 EdTech Student Engagement & Performance App")

# -------------------------------
# 1️⃣ Generate Synthetic Dataset
# -------------------------------

def generate_dataset(rows=1000):
    np.random.seed(42)

    course_ids = ['C101', 'C102', 'C103', 'C104']
    subjects = ['Math', 'Coding', 'HRM', 'Data Science']

    data = {
        "Student_ID": range(1, rows+1),
        "Course_ID": np.random.choice(course_ids, rows),
        "Subject": np.random.choice(subjects, rows),
        "Engagement_Hours": np.random.normal(20, 5, rows).round(2),
        "Assessment_Score": np.random.normal(70, 10, rows).round(2),
        "Login_Frequency": np.random.randint(5, 50, rows),
        "Assignment_Submission_Rate": np.random.uniform(40, 100, rows).round(2),
    }

    df = pd.DataFrame(data)

    # Completion logic
    df["Completion_Status"] = np.where(
        (df["Engagement_Hours"] > 18) & 
        (df["Assignment_Submission_Rate"] > 60),
        "Completed",
        "Dropped"
    )

    # Add messy email column (for regex)
    df["Email"] = df["Student_ID"].apply(lambda x: f"student{x}@edtech.com")

    # Introduce some missing values
    df.loc[np.random.choice(df.index, 20), "Engagement_Hours"] = np.nan

    return df


# -------------------------------
# 2️⃣ Upload or Generate
# -------------------------------

option = st.radio("Choose Data Source:", ["Generate Sample Dataset (1000 rows)", "Upload CSV"])

if option == "Generate Sample Dataset (1000 rows)":
    df = generate_dataset()
else:
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.stop()

st.subheader("🔎 Raw Data Preview")
st.dataframe(df.head())

# -------------------------------
# 3️⃣ Data Cleaning
# -------------------------------

st.subheader("🧹 Data Cleaning")

# Handle Missing Values
df["Engagement_Hours"].fillna(df["Engagement_Hours"].mean(), inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

st.write("Missing values handled & duplicates removed.")

# -------------------------------
# 4️⃣ Data Transformation
# -------------------------------

st.subheader("🔄 Data Transformation")

# Convert Completion to binary
df["Completion_Flag"] = df["Completion_Status"].map({"Completed": 1, "Dropped": 0})

# Normalize numeric features
scaler = MinMaxScaler()
num_cols = ["Engagement_Hours", "Assessment_Score", "Login_Frequency", "Assignment_Submission_Rate"]
df[num_cols] = scaler.fit_transform(df[num_cols])

st.write("Normalization applied using MinMaxScaler.")

# -------------------------------
# 5️⃣ Feature Engineering
# -------------------------------

st.subheader("⚙️ Feature Engineering")

# Engagement Score
df["Engagement_Score"] = (
    df["Engagement_Hours"] * 0.4 +
    df["Login_Frequency"] * 0.3 +
    df["Assignment_Submission_Rate"] * 0.3
)

# High Engagement Category
df["High_Engagement"] = np.where(df["Engagement_Score"] > 0.6, 1, 0)

st.write("New features created: Engagement_Score & High_Engagement")

# -------------------------------
# 6️⃣ Regular Expression
# -------------------------------

st.subheader("🔎 Regular Expression Processing")

# Extract domain from email
df["Email_Domain"] = df["Email"].apply(lambda x: re.findall(r'@(.+)', x)[0])

st.write("Extracted email domain using regex.")

# -------------------------------
# 7️⃣ Model Training (Optional)
# -------------------------------

st.subheader("🤖 Logistic Regression Model")

X = df[["Engagement_Hours", "Assessment_Score", "Login_Frequency", 
        "Assignment_Submission_Rate", "Engagement_Score"]]

y = df["Completion_Flag"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

st.success(f"Model Accuracy: {round(accuracy*100,2)}%")

# -------------------------------
# 8️⃣ Download Cleaned Data
# -------------------------------

st.subheader("⬇️ Download Processed Dataset")

csv = df.to_csv(index=False).encode('utf-8')
st.download_button("Download CSV", csv, "processed_dataset.csv", "text/csv")

st.dataframe(df.head())

st.markdown("---")
st.markdown("✅ App Complete: Upload → Clean → Transform → Normalize → Feature Engineer → Regex → Model")
