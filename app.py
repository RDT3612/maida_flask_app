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

# Load encoders
with open(ENCODERS_PATH) as f:
    encoders = json.load(f)

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
        # Map suppliers
        sup1 = request.form['Maida Supplier 1']
        sup2 = request.form['Maida Supplier 2']
        code1 = encoders['Maida Supplier 1'].get(sup1)
        code2 = encoders['Maida Supplier 2'].get(sup2)
        if code1 is None or code2 is None:
            return render_template('index.html',
                                   submitted=False,
                                   encoders=encoders,
                                   error="Unknown supplier.")
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
        'Maida Supplier 1':encoders['Maida Supplier 1'][request.form['Maida Supplier 1']],
        'Maida Supplier 2':encoders['Maida Supplier 2'][request.form['Maida Supplier 2']],
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


