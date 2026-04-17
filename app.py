from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, make_response
import sqlite3, hashlib, os, pickle, json, csv
from io import StringIO
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier   # deep learning model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

# PDF generation – optional. If you have not installed weasyprint, comment out the next line.
# from weasyprint import HTML

app = Flask(__name__)
app.secret_key = 'epipredict_collins_2025_secret'

BASE   = os.path.dirname(os.path.abspath(__file__))
DB     = os.path.join(BASE, 'epipredict.db')
MPATH  = os.path.join(BASE, 'model', 'knn.pkl')
SPATH  = os.path.join(BASE, 'model', 'scaler.pkl')
EPATH  = os.path.join(BASE, 'model', 'encoder.pkl')
TRAIN  = os.path.join(BASE, 'data', 'training.csv')
TEST   = os.path.join(BASE, 'data', 'testing.csv')

FEATURES = ['disease_enc','temperature_c','rainfall_mm','humidity_percent','population_density']
DISEASES  = ['Cholera','Influenza','Malaria','Typhoid']

# ──────────────────────────────────────────────
#  DATABASE (with timeout and autocommit)
# ──────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB, timeout=10, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    c = conn.cursor()
    c.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fullname TEXT NOT NULL,
            username TEXT UNIQUE NOT NULL,
            email    TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role     TEXT DEFAULT 'health_worker',
            created  TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            disease     TEXT,
            temperature REAL,
            rainfall    REAL,
            humidity    REAL,
            pop_density REAL,
            region      TEXT,                   -- new: rural community region
            result      INTEGER,
            label       TEXT,
            probability REAL,
            neighbors   TEXT,
            created     TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS model_runs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            accuracy  REAL,
            precision REAL,
            recall    REAL,
            f1        REAL,
            k         INTEGER,
            trained   TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            prediction_id INTEGER NOT NULL,
            is_correct INTEGER NOT NULL,
            comment TEXT,
            created TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE,
            UNIQUE(user_id, prediction_id)
        );
    ''')
    conn.commit(); conn.close()

def sha(pw): return hashlib.sha256(pw.encode()).hexdigest()

# ──────────────────────────────────────────────
#  MODEL (unchanged)
# ──────────────────────────────────────────────
def train_and_save():
    df_tr = pd.read_csv(TRAIN)
    df_te = pd.read_csv(TEST)

    df_tr = df_tr.dropna()
    df_te = df_te.dropna()

    le = LabelEncoder()
    le.fit(DISEASES)
    df_tr['disease_enc'] = le.transform(df_tr['disease'])
    df_te['disease_enc'] = le.transform(df_te['disease'])

    X_tr = df_tr[FEATURES].values
    y_tr = df_tr['outbreak'].values
    X_te = df_te[FEATURES].values
    y_te = df_te['outbreak'].values

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)

    knn = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')
    knn.fit(X_tr_s, y_tr)

    y_pred = knn.predict(X_te_s)
    acc  = float(accuracy_score(y_te, y_pred))
    prec = float(precision_score(y_te, y_pred, zero_division=0))
    rec  = float(recall_score(y_te, y_pred, zero_division=0))
    f1   = float(f1_score(y_te, y_pred, zero_division=0))
    cm   = confusion_matrix(y_te, y_pred).tolist()

    os.makedirs(os.path.join(BASE,'model'), exist_ok=True)
    pickle.dump(knn, open(MPATH,'wb'))
    pickle.dump(sc,  open(SPATH,'wb'))
    pickle.dump(le,  open(EPATH,'wb'))

    conn = get_db()
    conn.execute('INSERT INTO model_runs(accuracy,precision,recall,f1,k) VALUES(?,?,?,?,?)',
                 (acc,prec,rec,f1,7))
    conn.commit(); conn.close()
    return acc, prec, rec, f1, cm

def load_model():
    if not os.path.exists(MPATH):
        train_and_save()
    return (pickle.load(open(MPATH,'rb')),
            pickle.load(open(SPATH,'rb')),
            pickle.load(open(EPATH,'rb')))

def predict_one(disease, temp, rain, hum, pop):
    knn, sc, le = load_model()
    d_enc = le.transform([disease])[0]
    X = np.array([[d_enc, temp, rain, hum, pop]])
    Xs = sc.transform(X)
    pred      = int(knn.predict(Xs)[0])
    proba     = knn.predict_proba(Xs)[0]
    prob_out  = round(float(proba[1]) * 100, 2)
    dists, idxs = knn.kneighbors(Xs, n_neighbors=7)
    neighbors_info = [{'dist': round(float(d),4)} for d in dists[0]]
    return pred, prob_out, neighbors_info

# ──────────────────────────────────────────────
#  AUTH ROUTES
# ──────────────────────────────────────────────
@app.route('/')
def index():
    if 'uid' in session:
        return redirect(url_for('dashboard'))
    return render_template('home.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        fn   = request.form['fullname'].strip()
        un   = request.form['username'].strip()
        em   = request.form['email'].strip()
        pw   = request.form['password']
        cpw  = request.form['confirm_password']
        role = request.form.get('role','health_worker')
        if pw != cpw:
            flash('Passwords do not match.','error'); return render_template('register.html')
        if len(pw) < 6:
            flash('Password must be at least 6 characters.','error'); return render_template('register.html')
        try:
            conn = get_db()
            conn.execute('INSERT INTO users(fullname,username,email,password,role) VALUES(?,?,?,?,?)',
                         (fn, un, em, sha(pw), role))
            conn.commit(); conn.close()
            flash('Account created! Please log in.','success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists.','error')
    return render_template('register.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        un = request.form['username'].strip()
        pw = request.form['password']
        conn = get_db()
        u = conn.execute('SELECT * FROM users WHERE username=? AND password=?',
                         (un, sha(pw))).fetchone()
        conn.close()
        if u:
            session.update({'uid':u['id'],'uname':u['username'],
                            'fullname':u['fullname'],'role':u['role']})
            return redirect(url_for('dashboard'))
        flash('Invalid username or password.','error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear(); return redirect(url_for('login'))

# ──────────────────────────────────────────────
#  PROTECTED ROUTES
# ──────────────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*a,**kw):
        if 'uid' not in session: return redirect(url_for('login'))
        return f(*a,**kw)
    return wrapper

@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db()
    uid = session['uid']
    total   = conn.execute('SELECT COUNT(*) FROM predictions WHERE user_id=?',(uid,)).fetchone()[0]
    outbr   = conn.execute('SELECT COUNT(*) FROM predictions WHERE user_id=? AND result=1',(uid,)).fetchone()[0]
    safe    = conn.execute('SELECT COUNT(*) FROM predictions WHERE user_id=? AND result=0',(uid,)).fetchone()[0]

    # feedback stats
    total_feedback = conn.execute('SELECT COUNT(*) FROM feedback WHERE user_id=?', (uid,)).fetchone()[0]
    correct_feedback = conn.execute('SELECT COUNT(*) FROM feedback WHERE user_id=? AND is_correct=1', (uid,)).fetchone()[0]
    user_accuracy = (correct_feedback / total_feedback * 100) if total_feedback > 0 else None

    # recent predictions with feedback status
    recent = conn.execute('''
        SELECT p.*, f.is_correct as feedback_correct
        FROM predictions p
        LEFT JOIN feedback f ON p.id = f.prediction_id AND f.user_id = p.user_id
        WHERE p.user_id = ?
        ORDER BY p.id DESC
        LIMIT 6
    ''', (uid,)).fetchall()

    metrics = conn.execute('SELECT * FROM model_runs ORDER BY id DESC LIMIT 1').fetchone()
    d_stats = conn.execute(
        'SELECT disease, COUNT(*) cnt, SUM(result) outbreaks FROM predictions WHERE user_id=? GROUP BY disease',(uid,)
    ).fetchall()
    trend = conn.execute(
        "SELECT strftime('%Y-%m',created) mo, COUNT(*) cnt, SUM(result) outs "
        "FROM predictions WHERE user_id=? GROUP BY mo ORDER BY mo DESC LIMIT 6",(uid,)
    ).fetchall()
    conn.close()
    return render_template('dashboard.html',
                           total=total, outbr=outbr, safe=safe,
                           recent=recent, metrics=metrics,
                           d_stats=d_stats, trend=list(reversed(trend)),
                           total_feedback=total_feedback, user_accuracy=user_accuracy)

@app.route('/predict', methods=['GET','POST'])
@login_required
def predict():
    result = None
    if request.method == 'POST':
        try:
            disease = request.form['disease']
            temp    = float(request.form['temperature'])
            rain    = float(request.form['rainfall'])
            hum     = float(request.form['humidity'])
            pop     = float(request.form['pop_density'])
            region  = request.form.get('region', '').strip()  # new region field
            pred, prob, neighbors = predict_one(disease, temp, rain, hum, pop)
            label = 'OUTBREAK LIKELY' if pred==1 else 'NO OUTBREAK'

            local_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            conn = get_db()
            cur = conn.execute('''
                INSERT INTO predictions(
                    user_id, disease, temperature, rainfall, humidity, pop_density,
                    region, result, label, probability, neighbors, created
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (session['uid'], disease, temp, rain, hum, pop, region,
                  pred, label, prob, json.dumps(neighbors), local_time))
            pred_id = cur.lastrowid
            conn.commit(); conn.close()

            result = dict(id=pred_id, disease=disease, temp=temp, rain=rain,
                          hum=hum, pop=pop, region=region, pred=pred,
                          label=label, prob=prob, neighbors=neighbors)
        except Exception as e:
            flash(f'Prediction error: {str(e)}','error')
    return render_template('predict.html', result=result, diseases=DISEASES)

@app.route('/history')
@login_required
def history():
    conn = get_db()
    rows = conn.execute('''
        SELECT p.*, f.is_correct as feedback_correct
        FROM predictions p
        LEFT JOIN feedback f ON p.id = f.prediction_id AND f.user_id = p.user_id
        WHERE p.user_id = ?
        ORDER BY p.id DESC
    ''', (session['uid'],)).fetchall()
    conn.close()
    return render_template('history.html', predictions=rows)

# ──────────────────────────────────────────────
#  DELETE PREDICTION ROUTE
# ──────────────────────────────────────────────
@app.route('/delete_prediction/<int:pred_id>', methods=['POST'])
@login_required
def delete_prediction(pred_id):
    conn = get_db()
    pred = conn.execute('SELECT * FROM predictions WHERE id=? AND user_id=?',
                        (pred_id, session['uid'])).fetchone()
    if not pred:
        conn.close()
        flash('Prediction not found or access denied.', 'error')
        return redirect(url_for('history'))

    conn.execute('DELETE FROM predictions WHERE id=?', (pred_id,))
    conn.commit()
    conn.close()
    flash('Prediction deleted successfully.', 'success')
    return redirect(url_for('history'))

# ──────────────────────────────────────────────
#  FEEDBACK ROUTE
# ──────────────────────────────────────────────
@app.route('/feedback/<int:pred_id>', methods=['POST'])
@login_required
def give_feedback(pred_id):
    data = request.get_json()
    is_correct = data.get('is_correct')
    comment = data.get('comment', '')

    conn = get_db()
    # Verify ownership
    pred = conn.execute('SELECT * FROM predictions WHERE id=? AND user_id=?',
                        (pred_id, session['uid'])).fetchone()
    if not pred:
        conn.close()
        return jsonify({'error': 'Prediction not found'}), 404

    # Insert or replace feedback
    conn.execute('''
        INSERT INTO feedback (user_id, prediction_id, is_correct, comment)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(user_id, prediction_id) DO UPDATE SET
            is_correct = excluded.is_correct,
            comment = excluded.comment,
            created = CURRENT_TIMESTAMP
    ''', (session['uid'], pred_id, is_correct, comment))
    conn.commit()
    conn.close()

    return jsonify({'success': True})

# ──────────────────────────────────────────────
#  MAP ROUTE (outbreak map for rural communities)
# ──────────────────────────────────────────────
# Predefined coordinates for some rural regions (example for Kenya)
REGION_COORDINATES = {
    'Embu': (-0.537, 37.459),
    'Meru': (0.046, 37.649),
    'Machakos': (-1.521, 37.265),
    'Kitui': (-1.367, 38.010),
    'Kakamega': (0.284, 34.752),
    'Bungoma': (0.569, 34.560),
    'Busia': (0.460, 34.110),
    'Siaya': (0.062, 34.287),
    'Kisumu': (-0.102, 34.761),
    'Homa Bay': (-0.526, 34.456),
    'Migori': (-1.063, 34.473),
    'Kisii': (-0.677, 34.766),
    'Nyamira': (-0.566, 34.935),
    'Narok': (-1.083, 35.866),
    'Kajiado': (-1.850, 36.800),
    'Laikipia': (0.357, 36.783),
    'Nyeri': (-0.420, 36.950),
    'Kirinyaga': (-0.499, 37.283),
    'Murang\'a': (-0.721, 37.149),
    'Kiambu': (-1.170, 36.830)
}

@app.route('/outbreak_map')
@login_required
def outbreak_map():
    conn = get_db()
    # Fetch all predictions for the current user with region and result
    predictions = conn.execute('''
        SELECT id, disease, region, result, probability, label, created
        FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
    ''', (session['uid'],)).fetchall()
    conn.close()

    # Prepare data for the map: include coordinates if region known
    markers = []
    for p in predictions:
        if p['region'] and p['region'] in REGION_COORDINATES:
            lat, lon = REGION_COORDINATES[p['region']]
            markers.append({
                'lat': lat,
                'lon': lon,
                'disease': p['disease'],
                'probability': p['probability'],
                'result': p['result'],
                'label': p['label'],
                'region': p['region'],
                'date': p['created'][:10]
            })
    return render_template('map.html', markers=markers)

# ──────────────────────────────────────────────
#  REPORT ROUTES (CSV)
# ──────────────────────────────────────────────
@app.route('/report/csv')
@login_required
def report_csv():
    conn = get_db()
    rows = conn.execute('''
        SELECT id, disease, region, temperature, rainfall, humidity, pop_density,
               probability, result, label, created
        FROM predictions
        WHERE user_id = ?
        ORDER BY id DESC
    ''', (session['uid'],)).fetchall()
    conn.close()

    si = StringIO()
    cw = csv.writer(si)
    cw.writerow(['ID', 'Disease', 'Region', 'Temperature (°C)', 'Rainfall (mm)',
                 'Humidity (%)', 'Population Density', 'Outbreak Probability (%)',
                 'Result (1=Outbreak)', 'Label', 'Date'])
    for r in rows:
        cw.writerow([
            r['id'], r['disease'], r['region'], r['temperature'], r['rainfall'],
            r['humidity'], r['pop_density'], r['probability'],
            r['result'], r['label'], r['created']
        ])

    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=epipredict_report.csv"
    output.headers["Content-type"] = "text/csv"
    return output

# Optional PDF report – requires WeasyPrint. Uncomment if installed.
# (code omitted for brevity – you can copy from previous version)

# ──────────────────────────────────────────────
#  DEEP LEARNING MODEL COMPARISON ROUTE
# ──────────────────────────────────────────────
@app.route('/model_comparison')
@login_required
def model_comparison():
    # (same as before – no changes)
    df_tr = pd.read_csv(TRAIN)
    df_te = pd.read_csv(TEST)

    df_tr = df_tr.dropna()
    df_te = df_te.dropna()

    le = LabelEncoder()
    le.fit(DISEASES)
    df_tr['disease_enc'] = le.transform(df_tr['disease'])
    df_te['disease_enc'] = le.transform(df_te['disease'])

    X_tr = df_tr[FEATURES].values
    y_tr = df_tr['outbreak'].values
    X_te = df_te[FEATURES].values
    y_te = df_te['outbreak'].values

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    knn = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')
    knn.fit(X_tr_s, y_tr)
    y_pred_knn = knn.predict(X_te_s)

    mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu',
                        solver='adam', max_iter=500, random_state=42)
    mlp.fit(X_tr_s, y_tr)
    y_pred_mlp = mlp.predict(X_te_s)

    acc_knn = accuracy_score(y_te, y_pred_knn)
    prec_knn = precision_score(y_te, y_pred_knn, zero_division=0)
    rec_knn = recall_score(y_te, y_pred_knn, zero_division=0)
    f1_knn = f1_score(y_te, y_pred_knn, zero_division=0)
    cm_knn = confusion_matrix(y_te, y_pred_knn).tolist()

    acc_mlp = accuracy_score(y_te, y_pred_mlp)
    prec_mlp = precision_score(y_te, y_pred_mlp, zero_division=0)
    rec_mlp = recall_score(y_te, y_pred_mlp, zero_division=0)
    f1_mlp = f1_score(y_te, y_pred_mlp, zero_division=0)
    cm_mlp = confusion_matrix(y_te, y_pred_mlp).tolist()

    comparison = {
        'knn': {'accuracy': round(acc_knn*100,1), 'precision': round(prec_knn*100,1),
                'recall': round(rec_knn*100,1), 'f1': round(f1_knn*100,1),
                'confusion_matrix': cm_knn},
        'mlp': {'accuracy': round(acc_mlp*100,1), 'precision': round(prec_mlp*100,1),
                'recall': round(rec_mlp*100,1), 'f1': round(f1_mlp*100,1),
                'confusion_matrix': cm_mlp}
    }
    return render_template('model_comparison.html', comparison=comparison)

# ──────────────────────────────────────────────
#  TRAINING & API ROUTES
# ──────────────────────────────────────────────
@app.route('/train')
@login_required
def train_page():
    conn = get_db()
    metrics = conn.execute('SELECT * FROM model_runs ORDER BY id DESC LIMIT 1').fetchone()
    conn.close()
    return render_template('train.html', metrics=metrics)

@app.route('/retrain', methods=['POST'])
@login_required
def retrain():
    acc, prec, rec, f1, cm = train_and_save()
    return jsonify({'accuracy':round(acc*100,1),'precision':round(prec*100,1),
                    'recall':round(rec*100,1),'f1':round(f1*100,1),'cm':cm})

@app.route('/api/metrics')
@login_required
def api_metrics():
    conn = get_db()
    m = conn.execute('SELECT * FROM model_runs ORDER BY id DESC LIMIT 1').fetchone()
    conn.close()
    if not m: return jsonify({})
    return jsonify({'accuracy':round(m['accuracy']*100,1),'precision':round(m['precision']*100,1),
                    'recall':round(m['recall']*100,1),'f1':round(m['f1']*100,1)})

@app.route('/api/trend')
@login_required
def api_trend():
    conn = get_db()
    rows = conn.execute(
        "SELECT strftime('%Y-%m',created) mo, COUNT(*) cnt, COALESCE(SUM(result),0) outs "
        "FROM predictions WHERE user_id=? GROUP BY mo ORDER BY mo ASC LIMIT 12",(session['uid'],)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])

if __name__ == '__main__':
    init_db()
    if not os.path.exists(MPATH): train_and_save()
    app.run(debug=True, port=5000)