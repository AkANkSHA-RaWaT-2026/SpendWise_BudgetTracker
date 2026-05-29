"""
SpendWise - Flask Application
A complete budget tracking application with ML-powered categorization
"""

import os
import logging
import threading
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timezone
from collections import defaultdict
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)


_secret = os.environ.get('SECRET_KEY')
if not _secret:
    raise RuntimeError(
        "SECRET_KEY is not set. "
        "Copy .env.example to .env and fill in a strong random value.\n"
        "  python -c \"import secrets; print(secrets.token_hex(32))\""
    )
app.config['SECRET_KEY'] = _secret
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///budget.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
csrf = CSRFProtect(app)


limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# ---------------------------------------------------------------------------
# ML model loading
# ---------------------------------------------------------------------------

EXPECTED_MODEL_VERSION = "1.2"    
_model_lock = threading.RLock()
_model = None
MODEL_LOADED = False


def _load_model():
    """Load the ML model from disk."""
    global _model, MODEL_LOADED
    with _model_lock:
        try:
            bundle = joblib.load('category_model.pkl')

            if not isinstance(bundle, dict) or bundle.get('version') != EXPECTED_MODEL_VERSION:
                logger.warning(
                    "category_model.pkl version mismatch (expected %s, got %s). "
                    "Re-run train_model.py then restart.",
                    EXPECTED_MODEL_VERSION, bundle.get('version') if isinstance(bundle, dict) else 'unknown'
                )
                return
            _model = bundle['pipeline']
            MODEL_LOADED = True
            logger.info("ML model v%s loaded successfully.", EXPECTED_MODEL_VERSION)
        except FileNotFoundError:
            logger.warning("category_model.pkl not found — run train_model.py first. Using rule-based fallback.")
        except Exception as exc:
            logger.warning("ML model failed to load: %s. Using rule-based fallback.", exc)


_load_model()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now():
    """Return current UTC-aware datetime."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Database models
# ---------------------------------------------------------------------------

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), default=_now)

    transactions = db.relationship('Transaction', backref='user', lazy=True, cascade='all, delete-orphan')
    budgets = db.relationship('Budget', backref='user', lazy=True, cascade='all, delete-orphan')
    goals = db.relationship('Goal', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    type = db.Column(db.String(20), nullable=False)   # 'income' | 'expense'
    amount = db.Column(db.Float, nullable=False)
    category = db.Column(db.String(50), nullable=False)
    description = db.Column(db.String(200))
    date = db.Column(db.DateTime(timezone=True), default=_now)
    created_at = db.Column(db.DateTime(timezone=True), default=_now)


class Budget(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    month = db.Column(db.String(7), nullable=False)   # YYYY-MM
    created_at = db.Column(db.DateTime(timezone=True), default=_now)


class Goal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    target_amount = db.Column(db.Float, nullable=False)
    current_amount = db.Column(db.Float, default=0)
    deadline = db.Column(db.DateTime(timezone=True))
    created_at = db.Column(db.DateTime(timezone=True), default=_now)
    completed = db.Column(db.Boolean, default=False)


# ---------------------------------------------------------------------------
# Jinja2 filter
# ---------------------------------------------------------------------------

def format_indian_currency(amount):
    """Format a number in the Indian numbering system (lakhs / crores)."""
    decimal_part = ""
    amount_str = str(float(amount))
    if '.' in amount_str:
        decimal_part = "." + amount_str.split('.')[1][:2]

    s = str(int(amount))
    if len(s) <= 3:
        return s + decimal_part

    s_rev = s[::-1]
    result = s_rev[:3]
    s_rev = s_rev[3:]
    while s_rev:
        result += ',' + s_rev[:2]
        s_rev = s_rev[2:]

    return result[::-1] + decimal_part


app.jinja_env.filters['indian_currency'] = format_indian_currency


@login_manager.user_loader
def load_user(user_id):

    return db.session.get(User, int(user_id))


# ---------------------------------------------------------------------------
# Category prediction
# ---------------------------------------------------------------------------

def predict_category(description, transaction_type='expense'):
    """Return (category, confidence_pct) for a given transaction description."""
    with _model_lock:
        pipeline = _model
        loaded   = MODEL_LOADED

    if loaded and pipeline and description:
        try:
            prediction = pipeline.predict([description])[0]
            proba = pipeline.predict_proba([description])[0]

            if any(p < 0 for p in proba):
                raise ValueError("Negative predict_proba values detected -- check vectorizer config")
            return prediction, round(float(max(proba)) * 100, 2)
        except Exception as exc:
            logger.warning("ML prediction error: %s. Falling back to rules.", exc)

    desc = description.lower() if description else ""

    # ------------------------------------------------------------------ #
    # Income rule-based fallback (Indian keywords)                         #
    # ------------------------------------------------------------------ #
    if transaction_type == 'income':
        income_rules = {
            'Salary':             ['salary', 'salary credit', 'monthly salary', 'ctc', 'payroll',
                                   'wages', 'neft salary', 'employer transfer', 'salary transfer',
                                   'in hand salary', 'take home'],
            'Freelance':          ['freelance', 'consulting fee', 'client payment', 'project payment',
                                   'upwork', 'fiverr', 'toptal', 'consulting income', 'gig payment',
                                   'contract payment'],
            'Business Income':    ['business revenue', 'sales income', 'invoice payment', 'gst invoice',
                                   'customer payment', 'business transfer', 'shop income', 'trade income',
                                   'profit transfer', 'upi business'],
            'Rental Income':      ['rent received', 'house rent received', 'flat rent income',
                                   'tenant rent', 'rental income', 'property rent', 'pg income',
                                   'office rent received', 'shop rent received'],
            'Interest & Returns': ['interest credited', 'fd interest', 'savings interest', 'rd interest',
                                   'mutual fund returns', 'sip returns', 'nsc interest',
                                   'ppf interest', 'bank interest', 'bond interest', 'returns credited'],
            'Dividends':          ['dividend', 'stock dividend', 'share dividend', 'equity dividend',
                                   'zerodha dividend', 'groww dividend', 'nse dividend', 'bse dividend'],
            'Bonus':              ['bonus', 'performance bonus', 'incentive', 'joining bonus',
                                   'festival bonus', 'diwali bonus', 'annual bonus', 'appraisal bonus',
                                   'retention bonus', 'commission'],
            'Reimbursement':      ['reimbursement', 'expense reimbursement', 'travel reimbursement',
                                   'medical reimbursement', 'hr reimbursement', 'refund received',
                                   'claim settlement', 'insurance claim', 'gst refund'],
        }
        for category, keywords in income_rules.items():
            if any(kw in desc for kw in keywords):
                return category, 75.0
        return 'Other Income', 50.0

    # ------------------------------------------------------------------ #
    # Expense rule-based fallback (Indian keywords, original behaviour)    #
    # ------------------------------------------------------------------ #
    expense_rules = {
        'Groceries':       ['grocery', 'sabzi', 'kirana', 'dmart', 'reliance fresh', 'big bazaar',
                            'more supermarket', 'nature basket', 'milk', 'chicken', 'mutton',
                            'zepto', 'blinkit', 'bigbasket', 'swiggy instamart'],
        'Mobile Recharge': ['recharge', 'mobile recharge', 'airtel', 'jio', 'vi ', 'vodafone',
                            'bsnl', 'talktime', 'topup', 'prepaid'],
        'DTH Recharge':    ['dth', 'tata sky', 'tata play', 'dish tv', 'airtel digital',
                            'sun direct', 'd2h', 'videocon d2h', 'cable tv'],
        'Petrol':          ['petrol', 'diesel', 'fuel', 'hp petrol', 'bharat petroleum',
                            'indian oil', 'iocl', 'bpcl', 'hpcl', 'cng'],
        'Transportation':  ['uber', 'ola', 'rapido', 'auto rickshaw', 'bus fare', 'metro card',
                            'irctc', 'indigo', 'air india', 'makemytrip', 'redbus', 'yatra'],
        'Utilities':       ['electricity', 'bijli', 'water bill', 'gas bill', 'cylinder', 'lpg',
                            'bescom', 'mseb', 'tneb', 'uppcl'],
        'Internet':        ['internet', 'broadband', 'wifi', 'fiber', 'jio fiber',
                            'airtel xstream', 'act fibernet', 'hathway'],
        'Rent':            ['rent', 'house rent', 'flat rent', 'room rent', 'pg rent', 'hostel rent'],
        'EMI':             ['emi', 'loan emi', 'home loan', 'car loan', 'personal loan',
                            'hdfc loan', 'sbi loan', 'icici loan', 'bajaj finserv'],
        'Healthcare':      ['doctor', 'hospital', 'medical', 'pharmacy', 'medicine', 'apollo',
                            'fortis', 'max hospital', 'clinic', 'medplus', 'netmeds', 'pharmeasy', '1mg'],
        'Education':       ['school fees', 'college fees', 'tuition', 'coaching', 'byju',
                            'unacademy', 'vedantu', 'exam fee', 'course fee'],
        'Dining':          ['restaurant', 'zomato', 'swiggy', 'dominos', 'kfc', 'mcdonald',
                            'cafe', 'coffee', 'ccd', 'pizza hut', 'barbeque nation', 'haldiram'],
        'Entertainment':   ['netflix', 'amazon prime', 'hotstar', 'zee5', 'sonyliv', 'movie',
                            'pvr', 'inox', 'spotify', 'youtube premium', 'bookmyshow'],
        'Shopping':        ['amazon', 'flipkart', 'myntra', 'ajio', 'nykaa', 'meesho',
                            'tata cliq', 'snapdeal', 'reliance digital', 'croma', 'decathlon'],
        'Festival':        ['diwali', 'holi', 'eid', 'christmas', 'navratri', 'durga puja',
                            'wedding gift', 'rakhi', 'ganesh', 'pongal', 'onam'],
    }
    for category, keywords in expense_rules.items():
        if any(kw in desc for kw in keywords):
            return category, 75.0
    return 'Other', 50.0


# ---------------------------------------------------------------------------
# Stats / analytics helpers
# ---------------------------------------------------------------------------

def get_user_stats(user_id):
    transactions = Transaction.query.filter_by(user_id=user_id).all()
    total_income   = sum(t.amount for t in transactions if t.type == 'income')
    total_expenses = sum(t.amount for t in transactions if t.type == 'expense')
    return {
        'total_income': total_income,
        'total_expenses': total_expenses,
        'balance': total_income - total_expenses,
        'transaction_count': len(transactions),
    }


def get_category_spending(user_id, month=None):
    query = Transaction.query.filter_by(user_id=user_id, type='expense')
    if month:
        year, month_num = map(int, month.split('-'))
        query = query.filter(
            db.extract('year', Transaction.date) == year,
            db.extract('month', Transaction.date) == month_num,
        )
    totals = defaultdict(float)
    for t in query.all():
        totals[t.category] += t.amount
    return dict(totals)


def get_monthly_trend(user_id, months=6):
    """Return [(label, total), ...] sorted oldest to newest."""
    today = datetime.now()
    rows = []
    for i in range(months):
        month_num = today.month - i
        year_offset, month_num = divmod(month_num - 1, 12)
        month_num += 1
        year = today.year + year_offset
        label = datetime(year, month_num, 1).strftime('%b %Y')
        txns = Transaction.query.filter_by(user_id=user_id, type='expense').filter(
            db.extract('year', Transaction.date) == year,
            db.extract('month', Transaction.date) == month_num,
        ).all()
        rows.append(((year, month_num), label, sum(t.amount for t in txns)))

    # Sort chronologically — oldest first — by numeric (year, month) key
    rows.sort(key=lambda r: r[0])
    return [(label, total) for _key, label, total in rows]


def check_budget_alerts(user_id):
    current_month = datetime.now().strftime('%Y-%m')
    budgets = Budget.query.filter_by(user_id=user_id, month=current_month).all()
    spending = get_category_spending(user_id, current_month)
    alerts = []
    for b in budgets:
        spent = spending.get(b.category, 0)
        pct   = (spent / b.amount * 100) if b.amount > 0 else 0
        if pct >= 90:
            alerts.append({'category': b.category, 'spent': spent,
                           'budget': b.amount, 'percentage': pct})
    return alerts


def predict_next_month(user_id):
    """Predict next month spend based on 3-month average (excludes zero-spend months)."""
    trend    = get_monthly_trend(user_id, months=3)
    non_zero = [total for _, total in trend if total > 0]
    return round(sum(non_zero) / len(non_zero), 2) if non_zero else 0


def suggest_savings(user_id):
    balance = get_user_stats(user_id)['balance']
    return round(balance * 0.2, 2) if balance > 0 else 0


# ---------------------------------------------------------------------------
# Routes — Authentication
# ---------------------------------------------------------------------------

@app.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        confirm  = request.form.get('confirm_password', '')

        if not username or not email or not password:
            flash('All fields are required!', 'danger')
            return redirect(url_for('register'))
        if password != confirm:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('register'))
        if len(password) < 6:
            flash('Password must be at least 6 characters!', 'danger')
            return redirect(url_for('register'))
        if User.query.filter_by(username=username).first():
            flash('Username already exists!', 'danger')
            return redirect(url_for('register'))
        if User.query.filter_by(email=email).first():
            flash('Email already registered!', 'danger')
            return redirect(url_for('register'))

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
@limiter.limit("10 per minute")
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        flash('Invalid username or password!', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))


# ---------------------------------------------------------------------------
# Routes — Dashboard & Transactions
# ---------------------------------------------------------------------------

@app.route('/dashboard')
@login_required
def dashboard():
    stats                 = get_user_stats(current_user.id)
    recent_transactions   = Transaction.query.filter_by(user_id=current_user.id) \
                                .order_by(Transaction.date.desc()).limit(10).all()
    alerts                = check_budget_alerts(current_user.id)
    next_month_prediction = predict_next_month(current_user.id)
    savings_suggestion    = suggest_savings(current_user.id)
    goals                 = Goal.query.filter_by(user_id=current_user.id, completed=False).all()

    return render_template(
        'dashboard.html',
        stats=stats,
        transactions=recent_transactions,
        alerts=alerts,
        next_month_prediction=next_month_prediction,
        savings_suggestion=savings_suggestion,
        goals=goals,
        model_loaded=MODEL_LOADED,
    )


@app.route('/add_transaction', methods=['GET', 'POST'])
@login_required
def add_transaction():
    if request.method == 'POST':
        transaction_type = request.form.get('type', '')
        description      = request.form.get('description', '')
        category         = request.form.get('category', '')
        date_str         = request.form.get('date', '')


        if transaction_type not in ('income', 'expense'):
            flash('Invalid transaction type!', 'danger')
            return redirect(url_for('add_transaction'))

        try:
            amount = float(request.form.get('amount', ''))
            if amount <= 0:
                raise ValueError
        except (ValueError, TypeError):
            flash('Invalid amount!', 'danger')
            return redirect(url_for('add_transaction'))

        try:
            transaction_date = datetime.strptime(date_str, '%Y-%m-%d')
        except (ValueError, TypeError):
            transaction_date = datetime.now()

        db.session.add(Transaction(
            user_id=current_user.id, type=transaction_type,
            amount=amount, category=category,
            description=description, date=transaction_date,
        ))
        db.session.commit()
        flash('Transaction added successfully!', 'success')
        return redirect(url_for('dashboard'))

    predicted_category, confidence = 'Other', 0
    if request.args.get('description'):
        txn_type = request.args.get('type', 'expense')
        if txn_type not in ('income', 'expense'):
            txn_type = 'expense'
        predicted_category, confidence = predict_category(request.args['description'], txn_type)

    return render_template('add_transaction.html',
                           predicted_category=predicted_category,
                           confidence=confidence)


@app.route('/delete_transaction/<int:transaction_id>', methods=['POST'])
@login_required
def delete_transaction(transaction_id):

    transaction = db.session.get(Transaction, transaction_id)
    if transaction is None:
        flash('Transaction not found!', 'danger')
        return redirect(url_for('dashboard'))
    if transaction.user_id != current_user.id:
        flash('Unauthorized!', 'danger')
        return redirect(url_for('dashboard'))
    db.session.delete(transaction)
    db.session.commit()
    flash('Transaction deleted!', 'success')
    return redirect(url_for('dashboard'))


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.route('/api/category_spending')
@login_required
def api_category_spending():
    month = request.args.get('month')
    data  = get_category_spending(current_user.id, month)
    return jsonify({'categories': list(data.keys()), 'amounts': list(data.values())})


@app.route('/api/monthly_trend')
@login_required
def api_monthly_trend():

    try:
        months = max(1, min(int(request.args.get('months', 6)), 24))
    except (ValueError, TypeError):
        months = 6

    trend = get_monthly_trend(current_user.id, months)

    return jsonify({
        'months':  [label for label, _     in trend],
        'amounts': [total for _,     total in trend],
    })


@app.route('/api/predict_category', methods=['POST'])
@login_required
def api_predict_category():

    payload = request.get_json(silent=True) or {}
    transaction_type = payload.get('type', 'expense')
    if transaction_type not in ('income', 'expense'):
        transaction_type = 'expense'
    category, confidence = predict_category(payload.get('description', ''), transaction_type)
    return jsonify({'category': category, 'confidence': confidence})


# ---------------------------------------------------------------------------
# Routes — Budget management
# ---------------------------------------------------------------------------

@app.route('/budgets')
@login_required
def budgets():
    current_month = datetime.now().strftime('%Y-%m')
    user_budgets = Budget.query.filter_by(user_id=current_user.id, month=current_month).all()
    return render_template('budgets.html', budgets=user_budgets)


@app.route('/add_budget', methods=['POST'])
@login_required
def add_budget():
    category = request.form.get('category', '').strip()
    if not category:
        flash('Category is required!', 'danger')
        return redirect(url_for('budgets'))


    try:
        amount = float(request.form.get('amount', ''))
        if amount <= 0:
            raise ValueError
    except (ValueError, TypeError):
        flash('Invalid budget amount!', 'danger')
        return redirect(url_for('budgets'))

    month    = datetime.now().strftime('%Y-%m')
    existing = Budget.query.filter_by(user_id=current_user.id,
                                      category=category, month=month).first()
    if existing:
        existing.amount = amount
    else:
        db.session.add(Budget(user_id=current_user.id,
                              category=category, amount=amount, month=month))
    db.session.commit()
    flash('Budget updated!', 'success')
    return redirect(url_for('dashboard'))


# ---------------------------------------------------------------------------
# Routes — Goals management
# ---------------------------------------------------------------------------

@app.route('/add_goal', methods=['POST'])
@login_required
def add_goal():
    name = request.form.get('name', '').strip()
    if not name:
        flash('Goal name is required!', 'danger')
        return redirect(url_for('dashboard'))


    try:
        target_amount = float(request.form.get('target_amount', ''))
        if target_amount <= 0:
            raise ValueError
    except (ValueError, TypeError):
        flash('Invalid goal amount!', 'danger')
        return redirect(url_for('dashboard'))


    try:
        deadline_str = request.form.get('deadline', '')
        deadline = datetime.strptime(deadline_str, '%Y-%m-%d') if deadline_str else None
    except ValueError:
        deadline = None

    db.session.add(Goal(user_id=current_user.id, name=name,
                        target_amount=target_amount, deadline=deadline))
    db.session.commit()
    flash('Goal created!', 'success')
    return redirect(url_for('dashboard'))


@app.route('/update_goal/<int:goal_id>', methods=['POST'])
@login_required
def update_goal(goal_id):

    goal = db.session.get(Goal, goal_id)
    if goal is None:
        flash('Goal not found!', 'danger')
        return redirect(url_for('dashboard'))
    if goal.user_id != current_user.id:
        flash('Unauthorized!', 'danger')
        return redirect(url_for('dashboard'))


    try:
        amount = float(request.form.get('amount', ''))
        if amount <= 0:
            raise ValueError
    except (ValueError, TypeError):
        flash('Invalid amount!', 'danger')
        return redirect(url_for('dashboard'))

    goal.current_amount += amount
    if goal.current_amount >= goal.target_amount:
        goal.completed = True
    db.session.commit()
    flash('Goal updated!', 'success')
    return redirect(url_for('dashboard'))


# ---------------------------------------------------------------------------
# DB init & entry point
# ---------------------------------------------------------------------------

def init_db():
    with app.app_context():
        db.create_all()
        logger.info("Database initialised.")


if __name__ == '__main__':
    init_db()

    debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    port       = int(os.environ.get('PORT', 5000))
    logger.info("Starting on port %d (debug=%s, ML=%s)", port, debug_mode, MODEL_LOADED)
    app.run(debug=debug_mode, host='127.0.0.1', port=port)