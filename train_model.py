import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

MODEL_VERSION = "1.3"

# ---------------------------------------------------------------------------
# Expense training data — 16 categories
# ---------------------------------------------------------------------------

TRAINING_DATA = [
    ("dmart groceries purchase", "Groceries"),
    ("reliance fresh vegetables purchase", "Groceries"),
    ("big bazaar monthly ration shopping", "Groceries"),
    ("more supermarket grocery shopping", "Groceries"),
    ("nature basket fruits vegetables", "Groceries"),
    ("kirana store daily grocery items", "Groceries"),
    ("sabzi mandi vegetables weekly", "Groceries"),
    ("milk curd from local dairy", "Groceries"),
    ("chicken mutton from butcher shop", "Groceries"),
    ("zepto grocery instant delivery", "Groceries"),
    ("blinkit vegetables grocery order", "Groceries"),
    ("swiggy instamart grocery delivery", "Groceries"),
    ("bigbasket monthly grocery order", "Groceries"),
    ("grofers household grocery items", "Groceries"),
    ("supermarket weekly grocery shopping", "Groceries"),
    ("daily essentials grocery store", "Groceries"),
    ("rice dal atta grocery purchase", "Groceries"),
    ("fresh produce grocery market", "Groceries"),
    # — distinct from airtel fiber (Internet) and airtel dth (DTH Recharge)
    ("jio prepaid recharge 239 plan", "Mobile Recharge"),
    ("airtel mobile prepaid recharge 299", "Mobile Recharge"),
    ("airtel postpaid mobile bill payment", "Mobile Recharge"),
    ("vi talktime topup prepaid", "Mobile Recharge"),
    ("vodafone prepaid mobile recharge", "Mobile Recharge"),
    ("bsnl mobile prepaid recharge", "Mobile Recharge"),
    ("jio unlimited mobile plan 28 days", "Mobile Recharge"),
    ("airtel 28 days prepaid mobile plan", "Mobile Recharge"),
    ("vi unlimited calling mobile plan", "Mobile Recharge"),
    ("mobile topup paytm wallet", "Mobile Recharge"),
    ("phone recharge gpay upi", "Mobile Recharge"),
    ("jio monthly prepaid mobile", "Mobile Recharge"),
    ("vi monthly postpaid mobile bill", "Mobile Recharge"),
    ("prepaid mobile plan renewal sim", "Mobile Recharge"),
    ("sim card recharge online", "Mobile Recharge"),
    ("mobile data plan recharge", "Mobile Recharge"),
    ("talktime recharge 50 rupees", "Mobile Recharge"),
    ("phone bill prepaid top up", "Mobile Recharge"),
    ("tata sky monthly dth recharge", "DTH Recharge"),
    ("dish tv dth subscription renewal", "DTH Recharge"),
    ("airtel digital tv dth monthly", "DTH Recharge"),
    ("airtel dth south pack recharge", "DTH Recharge"),
    ("sun direct dth recharge 200", "DTH Recharge"),
    ("d2h monthly dth plan", "DTH Recharge"),
    ("videocon d2h dth recharge", "DTH Recharge"),
    ("tata play hd dth pack", "DTH Recharge"),
    ("cable tv operator monthly bill", "DTH Recharge"),
    ("dth plan renewal set top box", "DTH Recharge"),
    ("dish tv south regional pack", "DTH Recharge"),
    ("tata sky plus dth subscription", "DTH Recharge"),
    ("sun nxt dth annual plan", "DTH Recharge"),
    ("local cable tv monthly bill", "DTH Recharge"),
    ("set top box monthly recharge", "DTH Recharge"),
    ("satellite tv monthly subscription", "DTH Recharge"),
    ("dish tv new connection charges", "DTH Recharge"),
    ("tv channel pack monthly", "DTH Recharge"),
    ("hp petrol pump fuel fill", "Petrol"),
    ("bharat petroleum diesel fill", "Petrol"),
    ("indian oil petrol station fill", "Petrol"),
    ("shell petrol bandra pump", "Petrol"),
    ("iocl fuel station petrol", "Petrol"),
    ("bpcl petrol pump fill tank", "Petrol"),
    ("hpcl fuel petrol fill", "Petrol"),
    ("petrol 50 litres tank full", "Petrol"),
    ("diesel full tank fill", "Petrol"),
    ("cng filling station charges", "Petrol"),
    ("fuel for car monthly petrol", "Petrol"),
    ("bike petrol fill scooty", "Petrol"),
    ("nayara energy petrol pump", "Petrol"),
    ("petrol pump upi payment", "Petrol"),
    ("fuel charges highway pump", "Petrol"),
    ("vehicle fuel fill monthly", "Petrol"),
    ("petrol diesel fill station", "Petrol"),
    ("cng petrol hybrid fuel fill", "Petrol"),
    # Added strong transport context words so ticket alone doesn't confuse.
    ("uber cab ride booking", "Transportation"),
    ("ola cab ride city", "Transportation"),
    ("rapido bike taxi ride", "Transportation"),
    ("auto rickshaw fare ride", "Transportation"),
    ("dtc bus pass recharge delhi", "Transportation"),
    ("delhi metro smart card recharge", "Transportation"),
    ("irctc train ticket travel booking", "Transportation"),
    ("redbus intercity bus ticket booking", "Transportation"),
    ("indigo flight travel booking", "Transportation"),
    ("air india flight ticket travel", "Transportation"),
    ("makemytrip hotel travel booking", "Transportation"),
    ("yatra flight ticket travel", "Transportation"),
    ("meru cab service airport", "Transportation"),
    ("local train monthly travel pass", "Transportation"),
    ("bmtc bus card recharge travel", "Transportation"),
    ("cab ride travel city upi", "Transportation"),
    ("metro card balance recharge", "Transportation"),
    ("bus ticket intercity travel booking", "Transportation"),
    ("bescom electricity bill payment bengaluru", "Utilities"),
    ("mseb bijli electricity bill online", "Utilities"),
    ("tneb current electricity bill", "Utilities"),
    ("uppcl electricity bill payment", "Utilities"),
    ("water board monthly supply bill", "Utilities"),
    ("lpg cylinder gas booking online", "Utilities"),
    ("indane gas cylinder booking", "Utilities"),
    ("hp gas cylinder booking online", "Utilities"),
    ("bharat gas cylinder booking", "Utilities"),
    ("electricity bill this month paid", "Utilities"),
    ("power bill payment online portal", "Utilities"),
    ("bijli ka electricity bill", "Utilities"),
    ("water supply municipal bill", "Utilities"),
    ("gas connection monthly bill", "Utilities"),
    ("piped gas png monthly bill", "Utilities"),
    ("electric bill discom payment", "Utilities"),
    ("lpg gas refill home delivery", "Utilities"),
    ("electricity meter reading bill", "Utilities"),
    # bsnl broadband explicitly distinguished from bsnl mobile recharge
    ("jio fiber home broadband monthly plan", "Internet"),
    ("airtel xstream fiber broadband bill", "Internet"),
    ("airtel broadband home internet plan", "Internet"),
    ("act fibernet home broadband monthly", "Internet"),
    ("hathway broadband internet bill", "Internet"),
    ("tata play fiber broadband plan", "Internet"),
    ("bsnl broadband home internet monthly", "Internet"),
    ("you broadband internet bill payment", "Internet"),
    ("excitel fiber broadband plan", "Internet"),
    ("home wifi broadband monthly bill", "Internet"),
    ("broadband internet recharge monthly", "Internet"),
    ("internet broadband bill payment", "Internet"),
    ("fiber optic broadband plan renewal", "Internet"),
    ("jio home fiber internet plan", "Internet"),
    ("act 150mbps fiber monthly plan", "Internet"),
    ("wifi router broadband connection bill", "Internet"),
    ("home internet plan monthly payment", "Internet"),
    ("fiber internet broadband subscription", "Internet"),
    ("house rent monthly payment landlord", "Rent"),
    ("flat rent transfer to landlord", "Rent"),
    ("room rent pg accommodation monthly", "Rent"),
    ("apartment monthly rent payment", "Rent"),
    ("hostel room rent monthly", "Rent"),
    ("office space rent monthly", "Rent"),
    ("rent transfer to house owner", "Rent"),
    ("monthly rent paid neft transfer", "Rent"),
    ("rental payment upi landlord", "Rent"),
    ("pg accommodation monthly charges", "Rent"),
    ("paying guest monthly house rent", "Rent"),
    ("studio apartment monthly rent paid", "Rent"),
    ("1bhk rent payment monthly", "Rent"),
    ("2bhk flat monthly rent transfer", "Rent"),
    ("ghar ka kiraya monthly payment", "Rent"),
    ("house rent agreement monthly", "Rent"),
    ("room rent deposit advance", "Rent"),
    ("flat monthly rent upi transfer", "Rent"),
    ("hdfc home loan emi monthly", "EMI"),
    ("sbi car loan monthly emi payment", "EMI"),
    ("icici personal loan emi deduction", "EMI"),
    ("axis bank loan installment emi", "EMI"),
    ("bajaj finserv emi auto deduction", "EMI"),
    ("kotak mahindra loan emi monthly", "EMI"),
    ("monthly emi auto deducted bank", "EMI"),
    ("two wheeler bike loan emi", "EMI"),
    ("education loan emi payment bank", "EMI"),
    ("credit card emi payment monthly", "EMI"),
    ("hdfc home loan installment emi", "EMI"),
    ("home loan monthly emi transfer", "EMI"),
    ("personal loan repayment emi", "EMI"),
    ("bike loan emi monthly payment", "EMI"),
    ("consumer durable loan emi", "EMI"),
    ("car loan installment monthly emi", "EMI"),
    ("bank loan emi auto debit", "EMI"),
    ("loan repayment installment monthly", "EMI"),
    ("apollo hospital doctor consultation fee", "Healthcare"),
    ("fortis hospital diagnostic test charges", "Healthcare"),
    ("max hospital visit consultation charges", "Healthcare"),
    ("aiims doctor consultation visit", "Healthcare"),
    ("medplus pharmacy medicine purchase", "Healthcare"),
    ("netmeds medicine online delivery", "Healthcare"),
    ("pharmeasy medicine order online", "Healthcare"),
    ("1mg medicine order delivery", "Healthcare"),
    ("thyrocare blood test pathology", "Healthcare"),
    ("pathology lab blood test fee", "Healthcare"),
    ("dentist clinic tooth treatment fee", "Healthcare"),
    ("doctor hospital consultation charges", "Healthcare"),
    ("hospital admission treatment charges", "Healthcare"),
    ("medicine purchase chemist pharmacy", "Healthcare"),
    ("annual health checkup package", "Healthcare"),
    ("eye specialist optician consultation", "Healthcare"),
    ("pharmacy medical store purchase", "Healthcare"),
    ("health test lab diagnostic", "Healthcare"),
    # so amazon alone doesn't dominate the Shopping signal here
    ("school tuition term fees payment", "Education"),
    ("college university semester fees", "Education"),
    ("byju learning app subscription monthly", "Education"),
    ("unacademy online learning subscription fee", "Education"),
    ("vedantu online tuition class payment", "Education"),
    ("coaching institute tuition monthly fees", "Education"),
    ("tuition teacher monthly fees payment", "Education"),
    ("upsc ias coaching payment fees", "Education"),
    ("jee neet entrance coaching fees", "Education"),
    ("textbook stationery school purchase", "Education"),
    ("school stationery notebook purchase", "Education"),
    ("exam application registration fee", "Education"),
    ("online certificate course enrollment fee", "Education"),
    ("coursera udemy annual subscription", "Education"),
    ("skill development training course fee", "Education"),
    ("books study material purchase", "Education"),
    ("class notes printing stationery", "Education"),
    ("educational app monthly subscription", "Education"),
    ("zomato dinner food order delivery", "Dining"),
    ("swiggy lunch food delivery order", "Dining"),
    ("swiggy biryani chicken order", "Dining"),
    ("dominos pizza delivery order", "Dining"),
    ("kfc bucket meal fried chicken", "Dining"),
    ("mcdonalds burger happy meal", "Dining"),
    ("cafe coffee day ccd visit", "Dining"),
    ("starbucks coffee latte order", "Dining"),
    ("pizza hut family meal dine", "Dining"),
    ("barbeque nation dinner outing", "Dining"),
    ("haldiram snacks sweets purchase", "Dining"),
    ("local restaurant dinner food bill", "Dining"),
    ("cafe lunch outing food bill", "Dining"),
    ("hotel food restaurant bill payment", "Dining"),
    ("biryani restaurant food order", "Dining"),
    ("south indian breakfast restaurant", "Dining"),
    ("food delivery online order dinner", "Dining"),
    ("restaurant dining bill payment", "Dining"),
    # Added movie/event context to all cinema entries
    ("netflix monthly streaming subscription", "Entertainment"),
    ("amazon prime video annual subscription", "Entertainment"),
    ("disney plus hotstar premium streaming", "Entertainment"),
    ("zee5 streaming subscription monthly", "Entertainment"),
    ("sonyliv premium streaming plan", "Entertainment"),
    ("spotify music premium subscription", "Entertainment"),
    ("youtube premium monthly subscription", "Entertainment"),
    ("pvr cinema movie ticket booking", "Entertainment"),
    ("inox multiplex movie ticket booking", "Entertainment"),
    ("bookmyshow movie cinema ticket", "Entertainment"),
    ("bookmyshow event concert ticket", "Entertainment"),
    ("carnival cinemas movie ticket", "Entertainment"),
    ("voot select streaming subscription", "Entertainment"),
    ("mxplayer pro streaming plan", "Entertainment"),
    ("jiocinema premium streaming pass", "Entertainment"),
    ("apple tv plus streaming subscription", "Entertainment"),
    ("movie ticket online booking cinema", "Entertainment"),
    ("web series streaming subscription", "Entertainment"),
    
    ("amazon india product purchase order", "Shopping"),         
    ("amazon shopping delivery order", "Shopping"),
    ("flipkart big billion sale order", "Shopping"),
    ("myntra fashion clothing order", "Shopping"),
    ("ajio clothing fashion purchase", "Shopping"),
    ("nykaa beauty cosmetics order", "Shopping"),
    ("meesho clothing order delivery", "Shopping"),
    ("tata cliq product shopping", "Shopping"),
    ("snapdeal product purchase delivery", "Shopping"),
    ("reliance digital electronics purchase", "Shopping"),
    ("croma laptop mobile purchase", "Shopping"),
    ("decathlon sports equipment purchase", "Shopping"),
    ("lifestyle fashion store purchase", "Shopping"),
    ("westside clothing fashion purchase", "Shopping"),
    ("zudio fashion clothing shopping", "Shopping"),
    ("pantaloons clothing store purchase", "Shopping"),
    ("online shopping product delivery", "Shopping"),
    ("ecommerce purchase order delivered", "Shopping"),
    ("diwali gift shopping festival", "Festival"),
    ("holi colour celebration festival", "Festival"),
    ("eid special celebration purchase", "Festival"),
    ("christmas gifts family festival", "Festival"),
    ("navratri garba celebration event", "Festival"),
    ("durga puja donation festival", "Festival"),
    ("wedding gift purchase celebration", "Festival"),
    ("rakhi gift sister festival", "Festival"),
    ("ganesh chaturthi puja celebration", "Festival"),
    ("onam sadya festival expenses", "Festival"),
    ("pongal festival celebration", "Festival"),
    ("bhai dooj festival gift", "Festival"),
    ("new year celebration party", "Festival"),
    ("birthday party gift purchase", "Festival"),
    ("anniversary gift dinner celebration", "Festival"),
    ("festival season shopping celebration", "Festival"),
    ("pooja samagri festival purchase", "Festival"),
    ("festive occasion gift giving", "Festival"),
    
    ("gym fitness membership monthly fee", "Other"),
    ("salon haircut grooming charges", "Other"),
    ("laundry service monthly clothes", "Other"),
    ("dry cleaning clothes service", "Other"),
    ("pet dog cat food purchase vet", "Other"),
    ("veterinary vet pet clinic charges", "Other"),
    ("charity donation ngo trust", "Other"),
    ("newspaper magazine subscription monthly", "Other"),
    ("vehicle two wheeler insurance premium", "Other"),
    ("term life insurance premium annual", "Other"),
    ("home maintenance plumber repair", "Other"),
    ("electrician home repair charges", "Other"),
    ("bank account service charges fee", "Other"),
    ("courier parcel speed post service", "Other"),
    ("atm cash withdrawal transaction charges", "Other"),
    ("yoga fitness class monthly", "Other"),
    ("miscellaneous expense other charges", "Other"),
    ("personal care grooming salon", "Other"),
]

# ---------------------------------------------------------------------------
# Income training data — 9 categories 
# ---------------------------------------------------------------------------

INCOME_TRAINING_DATA = [
    # Salary
    ("monthly salary credited account", "Salary"),
    ("salary neft from employer", "Salary"),
    ("net salary transfer april", "Salary"),
    ("payroll credit from company", "Salary"),
    ("in hand salary this month", "Salary"),
    ("employer salary transfer done", "Salary"),
    ("wages credited bank account", "Salary"),
    ("ctc disbursement monthly", "Salary"),
    ("take home salary april", "Salary"),
    ("neft salary credit tcs", "Salary"),
    ("infosys salary payment", "Salary"),
    ("hdfc payroll salary credit", "Salary"),

    # Freelance
    ("upwork client payment received", "Freelance"),
    ("fiverr order completed payment", "Freelance"),
    ("freelance project payment upi", "Freelance"),
    ("consulting fee client transfer", "Freelance"),
    ("toptal monthly contractor payment", "Freelance"),
    ("freelance web development fee", "Freelance"),
    ("gig work payment received", "Freelance"),
    ("contract project milestone payment", "Freelance"),
    ("design consulting fee received", "Freelance"),
    ("remote freelance income credit", "Freelance"),
    ("client payment for app development", "Freelance"),
    ("content writing freelance payment", "Freelance"),

    # Business Income
    ("business sales income credit", "Business Income"),
    ("customer payment received upi", "Business Income"),
    ("shop daily sales collection", "Business Income"),
    ("gst invoice payment received", "Business Income"),
    ("business revenue transfer", "Business Income"),
    ("trade income monthly settlement", "Business Income"),
    ("online store sales income", "Business Income"),
    ("b2b client invoice cleared", "Business Income"),
    ("shop income deposit bank", "Business Income"),
    ("business profit transfer account", "Business Income"),
    ("amazon seller payout received", "Business Income"),
    ("flipkart seller weekly payout", "Business Income"),

    # Rental Income
    ("house rent received from tenant", "Rental Income"),
    ("flat rent received this month", "Rental Income"),
    ("pg rental income credited", "Rental Income"),
    ("shop rent received landlord", "Rental Income"),
    ("property rent transfer received", "Rental Income"),
    ("office space rent income", "Rental Income"),
    ("tenant monthly rent payment", "Rental Income"),
    ("rental income from flat", "Rental Income"),
    ("house rent upi from tenant", "Rental Income"),
    ("2bhk flat rent received", "Rental Income"),
    ("commercial property rent income", "Rental Income"),
    ("room rent received pg owner", "Rental Income"),

    # Interest & Returns
    ("fd interest credited account", "Interest & Returns"),
    ("savings account interest credit", "Interest & Returns"),
    ("rd maturity interest credited", "Interest & Returns"),
    ("ppf interest annual credit", "Interest & Returns"),
    ("nsc interest payment received", "Interest & Returns"),
    ("mutual fund sip returns", "Interest & Returns"),
    ("bank interest quarterly credit", "Interest & Returns"),
    ("bond interest payment received", "Interest & Returns"),
    ("fixed deposit returns credited", "Interest & Returns"),
    ("post office interest received", "Interest & Returns"),
    ("liquid fund returns credited", "Interest & Returns"),
    ("debt fund interest payment", "Interest & Returns"),

    # Dividends
    ("equity dividend zerodha credit", "Dividends"),
    ("stock dividend nse credited", "Dividends"),
    ("bse share dividend received", "Dividends"),
    ("groww mutual fund dividend", "Dividends"),
    ("upstox dividend credit account", "Dividends"),
    ("infosys stock dividend payment", "Dividends"),
    ("tcs quarterly dividend credit", "Dividends"),
    ("reliance industries dividend", "Dividends"),
    ("hdfc bank dividend received", "Dividends"),
    ("itc dividend credited demat", "Dividends"),
    ("kotak fund dividend payout", "Dividends"),
    ("sbi dividend payout received", "Dividends"),

    # Bonus
    ("performance bonus credited salary", "Bonus"),
    ("annual bonus from employer", "Bonus"),
    ("diwali bonus neft transfer", "Bonus"),
    ("joining bonus first month", "Bonus"),
    ("festival bonus october", "Bonus"),
    ("appraisal bonus increment credited", "Bonus"),
    ("retention bonus transfer", "Bonus"),
    ("sales incentive monthly credit", "Bonus"),
    ("commission payment credited", "Bonus"),
    ("target achievement bonus", "Bonus"),
    ("quarterly incentive payment", "Bonus"),
    ("project completion bonus", "Bonus"),

    # Reimbursement
    ("travel expense reimbursement hr", "Reimbursement"),
    ("medical reimbursement claim approved", "Reimbursement"),
    ("expense reimbursement from company", "Reimbursement"),
    ("insurance claim settlement credit", "Reimbursement"),
    ("gst input tax refund", "Reimbursement"),
    ("income tax refund received", "Reimbursement"),
    ("health insurance claim reimbursed", "Reimbursement"),
    ("vehicle repair reimbursement", "Reimbursement"),
    ("mobile bill reimbursement", "Reimbursement"),
    ("internet bill reimbursement", "Reimbursement"),
    ("fuel reimbursement from office", "Reimbursement"),
    ("lta reimbursement credited", "Reimbursement"),

    # Other Income
    ("cashback credited paytm wallet", "Other Income"),
    ("referral bonus credited app", "Other Income"),
    ("gift money received from family", "Other Income"),
    ("lottery prize amount credited", "Other Income"),
    ("scholarship amount credited account", "Other Income"),
    ("prize money competition won", "Other Income"),
    ("resale item sold amount received", "Other Income"),
    ("old phone sold online olx", "Other Income"),
    ("pocket money transfer parents", "Other Income"),
    ("miscellaneous income credited bank", "Other Income"),
    ("survey reward payment received", "Other Income"),
    ("affiliate marketing income payout", "Other Income"),
]

def generate_augmented_data():
    """Add UPI and NEFT prefix variants to training data."""
    all_data = TRAINING_DATA + INCOME_TRAINING_DATA
    augmented = []
    for desc, category in all_data:
        augmented.append((f"upi {desc}", category))
        augmented.append((f"{desc} payment", category))
        augmented.append((f"neft {desc}", category))
    return all_data + augmented

def train_model():
    print("Training SpendWise ML model...")

    data = generate_augmented_data()
    print(f"\nTotal samples : {len(data)}")
    print(f"Categories    : {len(set(c for _, c in data))}")

    df = pd.DataFrame(data, columns=['description', 'category'])
    print("\nCategory distribution:")
    print(df['category'].value_counts().to_string())

    X_train, X_test, y_train, y_test = train_test_split(
        df['description'], df['category'],
        test_size=0.2, random_state=42, stratify=df['category']
    )
    print(f"\nTraining : {len(X_train)}  |  Testing : {len(X_test)}")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=1500,      
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )),
        ('clf', MultinomialNB(alpha=0.3)),  
    ])

    print("\nTraining model ...")
    pipeline.fit(X_train, y_train)

    y_pred   = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy : {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    bundle = {
        'version':    MODEL_VERSION,
        'pipeline':   pipeline,
        'categories': sorted(df['category'].unique().tolist()),
    }
    joblib.dump(bundle, 'category_model.pkl')
    print(f"\nModel bundle v{MODEL_VERSION} saved as 'category_model.pkl'")

    # Smoke test — specifically targets the previously misclassifying cases
    test_cases = [
        ("airtel mobile prepaid recharge",       "Mobile Recharge"),
        ("airtel postpaid mobile bill",          "Mobile Recharge"),
        ("airtel xstream fiber broadband bill",  "Internet"),
        ("airtel broadband home plan",           "Internet"),
        ("airtel dth south pack",                "DTH Recharge"),
        ("airtel digital tv monthly",            "DTH Recharge"),
        ("swiggy lunch food delivery",           "Dining"),
        ("swiggy instamart grocery delivery",    "Groceries"),
        ("blinkit vegetables order",             "Groceries"),
        ("amazon product purchase order",        "Shopping"),
        ("amazon prime video subscription",      "Entertainment"),
        ("books study material purchase",        "Education"),
        ("bsnl mobile prepaid recharge",         "Mobile Recharge"),
        ("bsnl broadband home internet",         "Internet"),
        ("bookmyshow movie cinema ticket",       "Entertainment"),
        ("redbus intercity bus ticket booking",  "Transportation"),
        ("irctc train ticket booking",           "Transportation"),
        ("pet dog cat food vet",                 "Other"),
        ("jio fiber monthly plan",               "Internet"),
        ("netflix monthly subscription",         "Entertainment"),
        ("bescom electricity bill payment",      "Utilities"),
        ("apollo hospital consultation fee",     "Healthcare"),
        ("dmart groceries purchase",             "Groceries"),
        ("hdfc home loan emi",                   "EMI"),
        ("zomato dinner order",                  "Dining"),
        ("house rent monthly payment",           "Rent"),
        ("hp petrol pump fuel fill",             "Petrol"),
        ("diwali gift festival shopping",        "Festival"),
        ("monthly salary credited account",      "Salary"),
        ("upwork client payment received",       "Freelance"),
        ("house rent received from tenant",      "Rental Income"),
        ("fd interest credited account",         "Interest & Returns"),
        ("performance bonus credited salary",    "Bonus"),
        ("travel expense reimbursement hr",      "Reimbursement"),
        ("equity dividend zerodha credit",       "Dividends"),
        ("amazon seller payout received",        "Business Income"),
        ("cashback credited paytm wallet",       "Other Income"),
    ]

    print("\nSample predictions (expected → got):")
    
    correct = 0
    for desc, expected in test_cases:
        pred = pipeline.predict([desc])[0]
        conf = max(pipeline.predict_proba([desc])[0]) * 100
        status = "✓" if pred == expected else "✗"
        if pred == expected:
            correct += 1
        print(f"  {status}  '{desc}'")
        print(f"      Expected: {expected}  |  Got: {pred} ({conf:.1f}%)")

    print(f"\nSmoke test: {correct}/{len(test_cases)} correct")
    print("\n" + "=" * 55)
    print("Training complete.")