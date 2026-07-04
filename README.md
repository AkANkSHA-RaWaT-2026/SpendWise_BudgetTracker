# SpendWise – Budget Tracker

A web-based budget tracking app with ML-powered transaction categorisation, built for Indian UPI and bank transactions.

---

## What It Does

- Automatically predicts the expense/income category as you type a transaction description
- Tracks your monthly budget and alerts you when you're close to the limit
- Shows spending charts and monthly trends
- Includes a savings goal tracker with deadlines

---

## Tech Stack

**Backend**
- Python, Flask, SQLite

**Machine Learning**
- scikit-learn (Naive Bayes + TF-IDF) for category prediction

**Frontend**
- Chart.js for dashboard charts

**Auth & Security**
- Flask-Login, Flask-WTF

---

## ML Model

- Trained on ~300 manually labeled Indian transaction descriptions, augmented to ~900 samples across 25 categories
- Achieves 85% accuracy on held-out test data
- Falls back to keyword matching if the trained model fails to load, so the app degrades gracefully instead of breaking

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/AkANkSHA-RaWaT-2026/spendwise.git
cd spendwise
```

### 2. Create a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv venv
source venv/bin/activate
```

> Your terminal prompt should now show `(venv)` at the start — that confirms the virtual environment is active. Every `pip install` and `python` command below should be run **inside** this activated environment, not outside it.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root and set a secret key.

Generate a secure key using Python's `secrets` module (not `random`, which is predictable and unsuitable for security purposes):

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

This prints a random 64-character hex string. Copy it into your `.env` file:

```env
SECRET_KEY=paste_the_generated_value_here
```

`SECRET_KEY` is required by Flask-WTF/Flask-Login to sign session cookies and CSRF tokens securely — the app will not run safely without one.

> **Never commit this value to GitHub.** Generate it once per project (not per run, or existing user sessions will be invalidated), and keep it only in your local `.env` file, which should already be excluded via `.gitignore`.

### 5. Train the ML model

```bash
python train_model.py
```

This trains the Naive Bayes + TF-IDF classifier on the labeled transaction dataset and saves the model file the app loads at runtime.

### 6. Run the app

```bash
python app.py
```

### 7. Deactivate the environment when you're done

```bash
deactivate
```



