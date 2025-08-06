import os, json
from flask import Flask, request, render_template, redirect, url_for, flash
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import traceback


app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Add this for flash messages


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
    """Train model with proper error handling"""
    try:
        # Drop rows missing targets
        df_clean = df.dropna(subset=['Water added','ABC added'])
        
        if len(df_clean) == 0:
            raise ValueError("No valid training data available")
        
        X = df_clean[ALL_FEATURES]
        y = df_clean[['Water added','ABC added']]
        
        model = RandomForestRegressor(random_state=42)
        model.fit(X, y)
        
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        
        print(f"Model trained successfully with {len(df_clean)} samples")
        return model
    except Exception as e:
        print(f"Error training model: {str(e)}")
        traceback.print_exc()
        raise


# Load raw data
try:
    raw_df = pd.read_excel(DATA_PATH)
    raw_df.columns = raw_df.columns.str.strip()
    # Ensure feedback columns exist
    for col in ['Penetrometer Reading','Dough Temperature']:
        if col not in raw_df.columns:
            raw_df[col] = pd.NA
    # Initial categorical encoding for suppliers
    for col in ['Maida Supplier 1','Maida Supplier 2']:
        raw_df[col] = raw_df[col].astype('category').cat.codes
    print(f"Loaded raw data with {len(raw_df)} rows")
except Exception as e:
    print(f"Error loading raw data: {str(e)}")
    raw_df = pd.DataFrame()


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
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = train_model(raw_df) if len(raw_df) > 0 else None
else:
    model = train_model(raw_df) if len(raw_df) > 0 else None


@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        try:
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
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            traceback.print_exc()
            return render_template('index.html',
                                   submitted=False,
                                   encoders=encoders,
                                   error=f"Prediction error: {str(e)}")
    
    return render_template('index.html', submitted=False, encoders=encoders)


@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        print("Processing feedback...")
        
        # Get supplier names and update encoders if needed
        sup1 = request.form['Maida Supplier 1'].strip()
        sup2 = request.form['Maida Supplier 2'].strip()
        update_encoders(sup1, sup2)
        
        # Build record including feedback-only inputs
        record = {
            'AIA': float(request.form['AIA']),
            'TA': float(request.form['TA']),
            'Alcoholic Acidity': float(request.form['Alcoholic Acidity']),
            'Gluten': float(request.form['Gluten']),
            'SV': float(request.form['SV']),
            'Moisture': float(request.form['Moisture']),
            'Penetrometer Reading': float(request.form['Penetrometer Reading']),
            'Dough Temperature': float(request.form['Dough Temperature']),
            'Maida Supplier 1': encoders['Maida Supplier 1'][sup1],
            'Maida Supplier 2': encoders['Maida Supplier 2'][sup2],
            'Pred Water': float(request.form['Pred Water']),
            'Pred ABC': float(request.form['Pred ABC']),
            'Actual Water': float(request.form['Actual Water']),
            'Actual ABC': float(request.form['Actual ABC']),
            'Quality': request.form['Quality']
        }
        
        # Save feedback to CSV
        fb_df = pd.DataFrame([record])
        os.makedirs(os.path.dirname(FEEDBACK_CSV), exist_ok=True)
        
        if os.path.exists(FEEDBACK_CSV):
            fb_df.to_csv(FEEDBACK_CSV, mode='a', header=False, index=False)
        else:
            fb_df.to_csv(FEEDBACK_CSV, index=False)
        
        print("Feedback saved to CSV")
        
        # Prepare data for retraining
        try:
            # Read feedback data
            feedback_data = pd.read_csv(FEEDBACK_CSV)
            print(f"Loaded feedback data with {len(feedback_data)} rows")
            
            # Rename columns to match training data format
            feedback_data = feedback_data.rename(columns={
                'Actual Water': 'Water added',
                'Actual ABC': 'ABC added'
            })
            
            # Combine with original data
            if len(raw_df) > 0:
                # Ensure all columns are present in both dataframes
                for col in ALL_FEATURES + ['Water added', 'ABC added']:
                    if col not in feedback_data.columns:
                        feedback_data[col] = pd.NA
                    if col not in raw_df.columns:
                        raw_df[col] = pd.NA
                
                combined = pd.concat([raw_df, feedback_data], ignore_index=True, sort=False)
            else:
                combined = feedback_data.copy()
            
            print(f"Combined dataset has {len(combined)} rows")
            
            # Retrain model
            global model
            model = train_model(combined)
            
            flash('Feedback submitted successfully! Model has been retrained.', 'success')
            print("Model retrained successfully")
            
        except Exception as e:
            print(f"Error during model retraining: {str(e)}")
            traceback.print_exc()
            flash(f'Feedback saved, but model retraining failed: {str(e)}', 'warning')
        
    except Exception as e:
        print(f"Error processing feedback: {str(e)}")
        traceback.print_exc()
        flash(f'Error processing feedback: {str(e)}', 'error')
    
    return redirect(url_for('index'))


if __name__=='__main__':
    app.run(debug=True)



