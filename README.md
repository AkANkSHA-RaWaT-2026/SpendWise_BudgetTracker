SpendWise – Budget Tracker
A web-based budget tracking app with ML-powered transaction categorisation, built for Indian UPI and bank transactions.
Built as part of my B.Tech CSE (AI/ML) project at Aravali College of Engineering and Management, Faridabad (Jan–Jul 2026).

What it does:-
      Automatically predicts expense/income category as you type the transaction description
      Tracks your monthly budget and alerts when you're close to the limit
      Shows spending charts and monthly trends
      Savings goal tracker with deadlines

Tech used:
Python, Flask, SQLite
scikit-learn (Naive Bayes + TF-IDF) for category prediction
Chart.js for dashboard charts
Flask-Login, Flask-WTF for auth and security

ML model-
Trained on ~300 labeled Indian transaction descriptions, augmented to ~900 samples across 25 categories. Achieved 85% accuracy on test data. Falls back to keyword matching if the model isn't loaded.

How to run
bashpip install -r requirements.txt
python train_model.py
python app.py
Set a SECRET_KEY in your .env file before running.


Author
Akanksha Rawat
