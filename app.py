import os, json
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib


app = Flask(__name__)


# Paths
DATA_PATH     = os.path.join('data', 'ML-Process-control.xlsx')
MODEL_PATH    = os.path.join('models', 'water_abc_model.pkl')
ENCODERS_PATH = os.path.join('models', 'encoders.json')
FEEDBACK_CSV  = os.path.join('data', 'feedback.csv')


# Feature lists
ALL_FEATURES = [
    'AIA','TA','Alcoholic Acidity','Gluten','SV','Moisture',
    'Penetrometer Reading','Dough Temperature',
    'Maida Supplier 1','Maida Supplier 2'
]
PRED_FEATURES = [
    'AIA','TA','Alcoholic Acidity','Gluten','SV','Moisture',
    'Maida Supplier 1','Maida Supplier 2'
]


def update_encoders(supplier1, supplier2):
    """Update encoders with new supplier names if they don't exist"""
    global encoders
    
    # Check and add new suppliers to encoders
    if supplier1 not in encoders['Maida Supplier 1']:
        # Get the next available code
        max_code = max(encoders['Maida Supplier 1'].values()) if encoders['Maida Supplier 1'] else -1
        encoders['Maida Supplier 1'][supplier1] = max_code + 1
    
    if supplier2 not in encoders['Maida Supplier 2']:
        # Get the next available code
        max_code = max(encoders['Maida Supplier 2'].values()) if encoders['Maida Supplier 2'] else -1
        encoders['Maida Supplier 2'][supplier2] = max_code + 1
    
    # Save updated encoders
    os.makedirs(os.path.dirname(ENCODERS_PATH), exist_ok=True)
    with open(ENCODERS_PATH, 'w') as f:
        json.dump(encoders, f, indent=2)


def train_model(df: pd.DataFrame):
    # Drop rows missing targets
    df_clean = df.dropna(subset=['Water added','ABC added'])
    X = df_clean[ALL_FEATURES]
    y = df_clean[['Water added','ABC added']]
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


# Load raw data
raw_df = pd.read_excel(DATA_PATH)
raw_df.columns = raw_df.columns.str.strip()
# Ensure feedback columns exist
for col in ['Penetrometer Reading','Dough Temperature']:
    if col not in raw_df.columns:
        raw_df[col] = pd.NA
# Initial categorical encoding for suppliers
for col in ['Maida Supplier 1','Maida Supplier 2']:
    raw_df[col] = raw_df[col].astype('category').cat.codes


# Load or create encoders
if os.path.exists(ENCODERS_PATH):
    with open(ENCODERS_PATH) as f:
        encoders = json.load(f)
else:
    # Create default encoders if file doesn't exist
    encoders = {
        "Maida Supplier 1": {
            "Gharana": 0,
            "Gharana ": 1,
            "Shanti ": 2,
            "VNJ": 3
        },
        "Maida Supplier 2": {
            "Aahar": 0,
            "Gharana": 1,
            "VNJ": 2
        }
    }
    os.makedirs(os.path.dirname(ENCODERS_PATH), exist_ok=True)
    with open(ENCODERS_PATH, 'w') as f:
        json.dump(encoders, f, indent=2)


# Load or train model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = train_model(raw_df)


@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        # Parse only prediction inputs
        numeric_feats = ['AIA','TA','Alcoholic Acidity','Gluten','SV','Moisture']
        try:
            numeric_vals = [float(request.form[f]) for f in numeric_feats]
        except ValueError:
            return render_template('index.html',
                                   submitted=False,
                                   encoders=encoders,
                                   error="Enter valid numbers.")
        
        # Get supplier names from text input
        sup1 = request.form['Maida Supplier 1'].strip()
        sup2 = request.form['Maida Supplier 2'].strip()
        
        # Validate supplier names are not empty
        if not sup1 or not sup2:
            return render_template('index.html',
                                   submitted=False,
                                   encoders=encoders,
                                   error="Supplier names cannot be empty.")
        
        # Update encoders with new suppliers if needed
        update_encoders(sup1, sup2)
        
        # Get codes for suppliers
        code1 = encoders['Maida Supplier 1'][sup1]
        code2 = encoders['Maida Supplier 2'][sup2]
        
        # Predict
        vals = numeric_vals + [code1, code2]
        X_new = pd.DataFrame([vals], columns=PRED_FEATURES)
        w_pred, a_pred = model.predict(X_new)[0]
        w_pred, a_pred = round(w_pred,2), round(a_pred,2)
        return render_template('index.html',
                               submitted=True,
                               encoders=encoders,
                               water=w_pred,
                               abc=a_pred,
                               form=request.form)
    return render_template('index.html', submitted=False, encoders=encoders)


@app.route('/feedback', methods=['POST'])
def feedback():
    # Get supplier names and update encoders if needed
    sup1 = request.form['Maida Supplier 1'].strip()
    sup2 = request.form['Maida Supplier 2'].strip()
    update_encoders(sup1, sup2)
    
    # Build record including feedback-only inputs
    record = {
        'AIA':request.form['AIA'],
        'TA':request.form['TA'],
        'Alcoholic Acidity':request.form['Alcoholic Acidity'],
        'Gluten':request.form['Gluten'],
        'SV':request.form['SV'],
        'Moisture':request.form['Moisture'],
        'Penetrometer Reading':request.form['Penetrometer Reading'],
        'Dough Temperature':request.form['Dough Temperature'],
        'Maida Supplier 1':encoders['Maida Supplier 1'][sup1],
        'Maida Supplier 2':encoders['Maida Supplier 2'][sup2],
        'Pred Water':request.form['Pred Water'],
        'Pred ABC':request.form['Pred ABC'],
        'Actual Water':request.form['Actual Water'],
        'Actual ABC':request.form['Actual ABC'],
        'Quality':request.form['Quality']
    }
    fb_df = pd.DataFrame([record])
    os.makedirs(os.path.dirname(FEEDBACK_CSV), exist_ok=True)
    if os.path.exists(FEEDBACK_CSV):
        fb_df.to_csv(FEEDBACK_CSV, mode='a', header=False, index=False)
    else:
        fb_df.to_csv(FEEDBACK_CSV, index=False)
    
    # Retrain on combined data
    combined = pd.concat([raw_df, pd.read_csv(FEEDBACK_CSV)], sort=False)
    for col in ['Penetrometer Reading','Dough Temperature']:
        if col not in combined.columns:
            combined[col] = pd.NA
    for col in ['Maida Supplier 1','Maida Supplier 2']:
        combined[col] = combined[col].astype('category').cat.codes
    global model
    model = train_model(combined)
    return redirect(url_for('index'))


if __name__=='__main__':
    app.run(debug=True)



