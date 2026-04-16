# 💼 Job Market Analysis and Recommendation System

## 📌 Project Overview

This project analyzes job market trends and builds a smart recommendation system for job seekers. It uses real-world job posting data to identify high-demand roles, salary patterns, and emerging job categories.

The system also provides:

* 📊 Data-driven insights into job trends
* 🤖 Salary prediction using Machine Learning
* 🔍 Job recommendations based on user input

---

## 🎯 Objectives

* Analyze job market trends
* Identify high-demand job roles
* Compare salary patterns across categories and countries
* Build a recommendation system for job seekers
* Develop a predictive model for salary estimation

---

## 🧠 Key Features

### 📊 1. Exploratory Data Analysis (EDA)

* Salary distribution analysis
* Job category trends over time
* Country-wise salary comparison
* Monthly job posting trends
* Remote vs hourly job analysis

---

### 📈 2. Machine Learning Model

* Multiple models trained:

  * Linear Regression
  * Ridge & Lasso
  * Decision Tree
  * Random Forest
  * Gradient Boosting
* Best model selected automatically based on R² score
* Model saved using `pickle`

---

### 🔍 3. Job Recommendation System

* Built using **TF-IDF + Cosine Similarity**
* Suggests similar jobs based on title input
* Memory-efficient implementation (no full similarity matrix)

---

### 🌐 4. Streamlit Web App

* User-friendly interface
* Two main features:

  * Job Recommendation
  * Salary Prediction

---

## 🛠️ Technologies Used

* **Python**
* **Pandas, NumPy**
* **Matplotlib, Seaborn**
* **Scikit-learn**
* **Streamlit**
* **TF-IDF & Cosine Similarity**

---

## 📂 Project Structure

```
Job_Market_Analysis_and_Recommendation_System/
│
├── data/
│   └── raw/jobs.csv
│
├── models/
│   ├── best_model.pkl
│   └── model_columns.pkl
│
├── notebooks/
│   └── analysis.ipynb
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```
git clone <your-repo-link>
cd Job_Market_Analysis_and_Recommendation_System
```

### 2️⃣ Install dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run the application

```
streamlit run app.py
```

---

## 🚀 How It Works

### 🔹 Recommendation System

* User enters job title
* TF-IDF converts text into vectors
* Cosine similarity finds similar jobs

---

### 🔹 Salary Prediction

* User selects:

  * Job category
  * Country
  * Job type
* Model predicts expected salary

---

## 📊 Tasks Completed

✔ Task 1: Salary vs Job Categories Analysis
✔ Task 2: Emerging Job Categories Over Time
✔ Task 3: High-Demand Job Prediction
✔ Task 4: Salary Comparison by Country
✔ Task 5: Job Recommendation System
✔ Task 6: Monthly Job Market Trends
✔ Task 7: Remote Work Trend Analysis
✔ Task 8: Future Job Market Prediction

---

## 🧠 Key Learnings

* Data cleaning and preprocessing
* Feature engineering
* Handling missing values and outliers
* Model selection and evaluation
* Building scalable recommendation systems
* Deploying ML models using Streamlit

---

## 📌 Future Improvements

* Add real-time job data API
* Improve recommendation accuracy using NLP models
* Add interactive dashboards
* Deploy using Docker

---

## 👨‍💻 Author

**Devesh Kumar Mandal**

---

## 🌐 Live Application

👉 https://jobmarketanalysisandrecommendationsystem.streamlit.app

---

## 📜 License

This project is for educational and internship purposes.
