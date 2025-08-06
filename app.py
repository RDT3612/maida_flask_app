import os
import json
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

app = Flask(__name__)

# File paths
DATA_PATH     = os.path.join('data', 'ML-Process-control.xlsx')
MODEL_PATH    = os.path.join('models', 'water_abc_model.pkl')
ENCODERS_PATH = os.path.join('models', 'encoders.json')
FEEDBACK_CSV  = os.path.join('data', 'feedback.csv')

# All features used for training and prediction (must be consistent)
ALL_FEATURES = [
    'AIA', 'TA', 'Alcoholic Acidity', 'Gluten', 'SV', 'Moisture',
    'Penetrometer Reading', 'Dough Temperature',
    'Maida Supplier 1', 'Maida Supplier 2'
]

# Feature subset expected from user input for prediction (excluding feedback-only features)
PRED_FEATURES = [
    'AIA', 'TA', 'Alcoholic Acidity', 'Gluten', 'SV', 'Moisture',
    'Maida Supplier 1', 'Maida Supplier 2'
]

def update_encoders(supplier1, supplier2):
    """Add new supplier names with new codes if not present, and save encoders."""
    global encoders

    for key, supp in zip(['Maida Supplier 1', 'Maida Supplier 2'], [supplier1, supplier2]):
        if supp not in encoders[key]:
            max_code = max(encoders[key].values()) if encoders[key] else -1
            encoders[key][supp] = max_code + 1

    os.makedirs(os.path.dirname(ENCODERS_PATH), exist_ok=True)
    with open(ENCODERS_PATH, 'w') as f:
        json.dump(encoders, f, indent=2)

def train_model(df: pd.DataFrame):
    """Train RandomForestRegressor on the dataset after cleaning NA values."""
    # Drop rows missing targets
    df_clean = df.dropna(subset=['Water added', 'ABC added'])
    
    X = df_clean[ALL_FEATURES]
    y = df_clean[['Water added', 'ABC added']]

    # Drop rows with any missing feature values
    mask = X.notnull().all(axis=1)
    X = X[mask]
    y = y.loc[X.index]

    if len(X) == 0:
        raise ValueError("No valid rows with complete features/targets for training.")

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return model

def safe_float(val):
    """Convert input to float or pd.NA if conversion fails."""
    try:
        val_str = str(val).strip()
        if val_str == '':
            return pd.NA
        return float(val)
    except Exception:
        return pd.NA

# --- Initial setup on app startup ---

# Load raw Excel data
raw_df = pd.read_excel(DATA_PATH)
raw_df.columns = raw_df.columns.str.strip()

# Ensure feedback columns exist in raw data
for col in ['Penetrometer Reading', 'Dough Temperature']:
    if col not in raw_df.columns:
        raw_df[col] = pd.NA

# Encode suppliers in raw data
for col in ['Maida Supplier 1', 'Maida Supplier 2']:
    raw_df[col] = raw_df[col].astype('category').cat.codes

# Load or initialize encoders
if os.path.exists(ENCODERS_PATH):
    with open(ENCODERS_PATH) as f:
        encoders = json.load(f)
else:
    encoders = {
        "Maida Supplier 1": {"Gharana": 0, "Gharana ": 1, "Shanti ": 2, "VNJ": 3},
        "Maida Supplier 2": {"Aahar": 0, "Gharana": 1, "VNJ": 2}
    }
    os.makedirs(os.path.dirname(ENCODERS_PATH), exist_ok=True)
    with open(ENCODERS_PATH, 'w') as f:
        json.dump(encoders, f, indent=2)

# Load or train model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = train_model(raw_df)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Parse numeric features from form
        numeric_feats = ['AIA', 'TA', 'Alcoholic Acidity', 'Gluten', 'SV', 'Moisture']
        try:
            numeric_vals = [safe_float(request.form[f]) for f in numeric_feats]
            if any(val is pd.NA for val in numeric_vals):
                raise ValueError("Missing numerical input")
        except Exception:
            return render_template('index.html', submitted=False, encoders=encoders,
                                   error="Enter valid numbers for all inputs.", form=request.form)

        sup1 = request.form['Maida Supplier 1'].strip()
        sup2 = request.form['Maida Supplier 2'].strip()
        if not sup1 or not sup2:
            return render_template('index.html', submitted=False, encoders=encoders,
                                   error="Supplier names cannot be empty.", form=request.form)

        update_encoders(sup1, sup2)
        code1 = encoders['Maida Supplier 1'][sup1]
        code2 = encoders['Maida Supplier 2'][sup2]

        # For prediction, add zeros for Penetrometer Reading & Dough Temperature
        vals = numeric_vals + [0.0, 0.0, code1, code2]
        X_new = pd.DataFrame([vals], columns=ALL_FEATURES)

        try:
            w_pred, a_pred = model.predict(X_new)[0]
        except Exception as e:
            return render_template('index.html', submitted=False, encoders=encoders,
                                   error=f"Prediction error: {e}", form=request.form)

        w_pred, a_pred = round(w_pred, 2), round(a_pred, 2)

        return render_template('index.html', submitted=True, encoders=encoders,
                               water=w_pred, abc=a_pred, form=request.form)
    else:
        return render_template('index.html', submitted=False, encoders=encoders)

@app.route('/feedback', methods=['POST'])
def feedback():
    sup1 = request.form['Maida Supplier 1'].strip()
    sup2 = request.form['Maida Supplier 2'].strip()
    update_encoders(sup1, sup2)

    record = {
        'AIA': safe_float(request.form['AIA']),
        'TA': safe_float(request.form['TA']),
        'Alcoholic Acidity': safe_float(request.form['Alcoholic Acidity']),
        'Gluten': safe_float(request.form['Gluten']),
        'SV': safe_float(request.form['SV']),
        'Moisture': safe_float(request.form['Moisture']),
        'Penetrometer Reading': safe_float(request.form.get('Penetrometer Reading', '')),
        'Dough Temperature': safe_float(request.form.get('Dough Temperature', '')),
        'Maida Supplier 1': encoders['Maida Supplier 1'][sup1],
        'Maida Supplier 2': encoders['Maida Supplier 2'][sup2],
        'Pred Water': safe_float(request.form['Pred Water']),
        'Pred ABC': safe_float(request.form['Pred ABC']),
        'Actual Water': safe_float(request.form['Actual Water']),
        'Actual ABC': safe_float(request.form['Actual ABC']),
        'Quality': request.form.get('Quality', '').strip()
    }

    fb_df = pd.DataFrame([record])

    os.makedirs(os.path.dirname(FEEDBACK_CSV), exist_ok=True)
    if os.path.exists(FEEDBACK_CSV):
        fb_df.to_csv(FEEDBACK_CSV, mode='a', header=False, index=False)
    else:
        fb_df.to_csv(FEEDBACK_CSV, index=False)

    combined = pd.concat([raw_df, pd.read_csv(FEEDBACK_CSV)], sort=False)

    for col in ['Penetrometer Reading', 'Dough Temperature']:
        if col not in combined.columns:
            combined[col] = pd.NA

    # Ensure supplier columns are numeric codes (integers), filling missing with -1
    for col in ['Maida Supplier 1', 'Maida Supplier 2']:
        combined[col] = pd.to_numeric(combined[col], errors='coerce').fillna(-1).astype(int)

    # Retrain model on updated data
    try:
        global model
        model = train_model(combined)
    except ValueError:
        # If no valid data after filtering, skip retraining
        pass

    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)



