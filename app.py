#!/usr/bin/env python
# coding: utf-8

# In[5]:


import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import sqlite3
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import os

# App config
st.set_page_config(page_title="Intelligent Task Management System", page_icon="✅", layout="wide")

# Download NLTK stopwords
try:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
except Exception as e:
    st.error(f"Error downloading NLTK stopwords: {e}")
    st.stop()


# Custom CSS styling
st.markdown("""
    <style>
    html, body {
        background-color: #121212;
        color: #f0f0f0;
    }
    .main-container {
        padding: 2rem;
        background-color: #1e1e1e;
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.6);
        margin-top: 1rem;
    }
    .card {
        background-color: #2b2b2b;
        padding: 1rem;
        border-left: 5px solid #4dd0e1;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .header {
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-top: 10px;
        color: #4dd0e1;
    }
    .subheader {
        font-size: 24px;
        font-weight: bold;
        margin-top: 2rem;
        color: #81d4fa;
    }
    .task-title {
        font-size: 20px;
        font-weight: bold;
        color: #ffffff;
    }
    .stTextInput > div > div > input, .stTextArea > div > textarea, .stDateInput > div > input {
        background-color: #333;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)




# Title and description
st.markdown('<div class="header"> AI-Powered Task Management System </div>', unsafe_allow_html=True)
st.markdown("""
This application uses machine learning to classify, prioritize, and assign tasks based on their description, deadline, user workload, and expertise match.
Enter task details to get predictions and manage tasks efficiently.
""")


# Load models and preprocessors
@st.cache_resource
def load_models():
    model_files = [
        "priority_model.pkl",
        "category_model.pkl",
        "user_model.pkl",
        "tfidf_vectorizer.pkl",
        "le_priority.pkl",
        "le_category.pkl",
        "le_user.pkl"
    ]
    for file in model_files:
        if not os.path.exists(file):
            st.error(f"Error: '{file}' not found in {os.getcwd()}. Please ensure all model files are present.")
            return None, None, None, None, None, None, None
    try:
        priority_model = joblib.load("priority_model.pkl")
        category_model = joblib.load("category_model.pkl")
        user_model = joblib.load("user_model.pkl")
        tfidf = joblib.load("tfidf_vectorizer.pkl")
        le_priority = joblib.load("le_priority.pkl")
        le_category = joblib.load("le_category.pkl")
        le_user = joblib.load("le_user.pkl")
        return priority_model, category_model, user_model, tfidf, le_priority, le_category, le_user
    except Exception as e:
        st.error(f"Error loading models or preprocessors: {e}")
        return None, None, None, None, None, None, None


priority_model, category_model, user_model, tfidf, le_priority, le_category, le_user = load_models()

# Check if models loaded successfully
if None in (priority_model, category_model, user_model, tfidf, le_priority, le_category, le_user):
    st.stop()

# Connect to SQLite database
conn = sqlite3.connect('task_db.sqlite', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        description TEXT,
        processed_description TEXT,
        predicted_priority TEXT,
        predicted_category TEXT,
        assigned_user TEXT,
        timestamp TEXT,
        deadline TEXT,
        user_workload REAL,
        user_expertise_match INTEGER,
        status TEXT DEFAULT 'Pending'
    )
''')
conn.commit()


# Preprocessing function
def preprocess(text):
    return " ".join(
        [stemmer.stem(word) for word in text.lower().split()
         if word.isalpha() and word not in stop_words]
    )


# TaskManagementSystem class
class TaskManagementSystem:
    def __init__(self, priority_model, category_model, user_model, tfidf, le_priority, le_category, le_user):
        self.priority_model = priority_model
        self.category_model = category_model
        self.user_model = user_model
        self.tfidf = tfidf
        self.le_priority = le_priority
        self.le_category = le_category
        self.le_user = le_user

    def predict_task(self, task_description, deadline, user_workload, user_expertise_match):
        try:
            # Preprocess input
            processed_description = preprocess(task_description)
            task_vector = self.tfidf.transform([processed_description]).toarray()
            days_to_deadline = (datetime.strptime(deadline, '%Y-%m-%d') - datetime.now()).days
            features = np.hstack([task_vector, [[days_to_deadline, user_workload, user_expertise_match]]])

            # Predict
            priority = self.le_priority.inverse_transform(self.priority_model.predict(features))[0]
            category = self.le_category.inverse_transform(self.category_model.predict(features))[0]
            user = self.le_user.inverse_transform(self.user_model.predict(features))[0]

            # Debug: Show TF-IDF features and user prediction probabilities
            active_features = self.tfidf.get_feature_names_out()[task_vector[0] > 0]
            st.write("Active TF-IDF Features:", active_features.tolist())
            if hasattr(self.user_model, 'predict_proba'):
                user_probs = self.user_model.predict_proba(features)[0]
                user_prob_dict = {self.le_user.inverse_transform([i])[0]: prob for i, prob in enumerate(user_probs)}
                st.write("User Prediction Probabilities:", user_prob_dict)

            return {
                "task_description": task_description,
                "processed_description": processed_description,
                "priority": priority,
                "category": category,
                "assigned_user": user,
                "days_to_deadline": days_to_deadline,
                "user_workload": user_workload,
                "user_expertise_match": user_expertise_match
            }
        except ValueError as e:
            st.error(f"Error in prediction: {e}")
            return None


# Initialize the system
task_system = TaskManagementSystem(priority_model, category_model, user_model, tfidf, le_priority, le_category, le_user)

# Task Addition Section
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="subheader">🆕 Add a New Task</div>', unsafe_allow_html=True)

with st.form("task_form"):
    task_description = st.text_area("📝 Task Description", placeholder="e.g., Data analysis on sales data")
    deadline = st.date_input("📆 Deadline", min_value=date.today())
    user_workload = st.slider("👷 User Workload (0 to 1)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
    user_expertise_match = st.selectbox("🎯 User Expertise Match", [1, 0],
                                        format_func=lambda x: "Yes" if x == 1 else "No")
    submit_button = st.form_submit_button("🚀 Perform Task", help="Submit to predict task details")

    if submit_button:
        if not task_description.strip():
            st.error("Please enter a task description.")
        elif deadline < date.today():
            st.error("Deadline must be in the future.")
        else:
            # Make prediction
            result = task_system.predict_task(
                task_description=task_description,
                deadline=deadline.strftime('%Y-%m-%d'),
                user_workload=user_workload,
                user_expertise_match=user_expertise_match
            )

            if result:
                # Save to database
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute('''
                    INSERT INTO tasks (description, processed_description, predicted_priority, predicted_category, assigned_user, timestamp, deadline, user_workload, user_expertise_match)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result['task_description'],
                    result['processed_description'],
                    result['priority'],
                    result['category'],
                    result['assigned_user'],
                    timestamp,
                    deadline.strftime('%Y-%m-%d'),
                    result['user_workload'],
                    result['user_expertise_match']
                ))
                conn.commit()

                st.header("Prediction Results")
                st.success(
                    f"✅ Task assigned to **{result['assigned_user']}** with **{result['priority']}** priority and **{result['category']}** category.")
                st.json(result)
                st.markdown(f"""
                **Task Description**: {result['task_description']}  
                **Priority**: {result['priority']}  
                **Category**: {result['category']}  
                **Assigned User**: {result['assigned_user']}  
                **Deadline**: {deadline.strftime('%Y-%m-%d')}  
                **User Workload**: {result['user_workload']}  
                **User Expertise Match**: {"Yes" if result['user_expertise_match'] == 1 else "No"}
                """)
st.markdown('</div>', unsafe_allow_html=True)

# Task Viewer Section
st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.markdown('<div class="subheader">📋 View Tasks</div>', unsafe_allow_html=True)

user_filter = st.text_input("👤 Enter your username to view tasks (e.g., Alice, Bob, Charlie, Diana, Eve)")

if user_filter:
    filter_opt = st.radio("🔍 Filter", ["Pending", "Completed", "All"], horizontal=True)

    query = "SELECT * FROM tasks WHERE assigned_user = ?"
    if filter_opt == "Pending":
        query += " AND status = 'Pending'"
    elif filter_opt == "Completed":
        query += " AND status = 'Completed'"
    query += '''
        ORDER BY 
            CASE predicted_priority
                WHEN 'High' THEN 1
                WHEN 'Medium' THEN 2
                ELSE 3
            END, deadline ASC
    '''
    user_tasks = pd.read_sql_query(query, conn, params=(user_filter,))

    if user_tasks.empty:
        st.info(f"No tasks found for user '{user_filter}' with status '{filter_opt}'.")
    else:
        for _, row in user_tasks.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="card">
                    <div class="task-title">📝 {row['description']}</div>
                    📅 Deadline: <b>{row['deadline']}</b><br>
                    🔥 Priority: <b>{row['predicted_priority']}</b><br>
                    🏷️ Category: <b>{row['predicted_category']}</b><br>
                    👤 Assigned User: <b>{row['assigned_user']}</b><br>
                    ✅ Status: {"✅ Completed" if row['status'] == 'Completed' else "❌ Pending"}
                </div>
                """, unsafe_allow_html=True)

                if row['status'] != 'Completed':
                    if st.checkbox("✔️ Mark as Completed", key=f"complete_{row['id']}"):
                        cursor.execute("UPDATE tasks SET status = 'Completed' WHERE id = ?", (row['id'],))
                        conn.commit()
                        st.success("🎉 Task marked as completed.")
                        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# All Tasks Section
st.markdown('<div class="main-container">', unsafe_allow_html=True)
with st.expander("📦 Show All Tasks"):
    all_tasks = pd.read_sql_query("SELECT * FROM tasks ORDER BY timestamp DESC", conn)
    if all_tasks.empty:
        st.info("No tasks available.")
    else:
        st.dataframe(all_tasks)
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
# Close database connection
conn.close()


# In[ ]:




