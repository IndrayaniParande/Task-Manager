# AI-Powered Task Management System

This is a Streamlit-based intelligent task management system that utilizes machine learning to automate and optimize task classification, prioritization, and user assignment.

## Features
- Predicts task category using a trained classification model
- Determines task priority based on content and metadata
- Assigns tasks to the most suitable user considering workload and expertise
- Interactive web interface for adding and tracking tasks
- Task data stored in a persistent SQLite database

## Technologies Used
- Python
- Streamlit
- scikit-learn
- NLTK
- SQLite
- Joblib

## Project Structure
```
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── *.pkl                   # Pre-trained models and encoders
├── task_db.sqlite          # SQLite database file
```

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Launch the Streamlit application:
```bash
streamlit run app.py
```

3. Ensure the following model files are present in the same directory:
- priority_model.pkl
- category_model.pkl
- user_model.pkl
- tfidf_vectorizer.pkl
- le_priority.pkl
- le_category.pkl
- le_user.pkl

## Notes
- The system uses TF-IDF vectorization and scikit-learn models to make predictions.
- NLTK is used for basic preprocessing, including stopword removal and stemming.
- The application connects to a SQLite database (`task_db.sqlite`) to store and manage tasks.

## License
This project is licensed under the MIT License.

## Author
Indrayani Parande
