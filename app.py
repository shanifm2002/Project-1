import os
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
import sqlite3
from flask_bcrypt import Bcrypt

# ------------------------------
# Flask app setup
# ------------------------------
app = Flask(__name__, static_folder="../frontend")
CORS(app)
bcrypt = Bcrypt(app)

# ------------------------------
# SQLite user authentication setup
# ------------------------------
DB_PATH = "users.db"

def get_db():
    db = getattr(g, "_database", None)
    if db is None:
        db = g._database = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

def init_db():
    """Create SQLite database and users table automatically if not exists"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ------------------------------
# Register route
# ------------------------------
@app.route("/register", methods=["POST"])
def register():
    data = request.get_json()
    email = data.get("email")
    username = data.get("username")
    password = data.get("password")

    if not email or not username or not password:
        return jsonify({"error": "All fields are required"}), 400

    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM users WHERE email=? OR username=?", (email, username))
    existing = cur.fetchone()
    if existing:
        return jsonify({"error": "User already exists"}), 400

    hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")
    cur.execute("INSERT INTO users (email, username, password_hash) VALUES (?, ?, ?)",
                (email, username, hashed_pw))
    db.commit()
    return jsonify({"message": "User registered successfully"}), 201

# ------------------------------
# Login route
# ------------------------------
@app.route("/login", methods=["POST"])
def login():
    data = request.get_json()
    email_or_username = data.get("email_or_username")
    password = data.get("password")

    if not email_or_username or not password:
        return jsonify({"error": "All fields are required"}), 400

    db = get_db()
    cur = db.cursor()
    cur.execute("SELECT * FROM users WHERE email=? OR username=?", (email_or_username, email_or_username))
    user = cur.fetchone()

    if user and bcrypt.check_password_hash(user["password_hash"], password):
        return jsonify({
            "message": "Login successful",
            "user": {"id": user["id"], "username": user["username"], "email": user["email"]}
        }), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401


# ------------------------------
# Your original Inflation ML code (unchanged below)
# ------------------------------
MODELS_DIR = "saved_models"
DATA_FILE = "data/inflation_yearly_dataset.csv"  # Your CSV with all columns
os.makedirs(MODELS_DIR, exist_ok=True)

# In-memory store
models = {}
country_data = {}

# ------------------------------
# Serve frontend files
# ------------------------------
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if path == "":
        return send_from_directory(app.static_folder, "login.html")
    return send_from_directory(app.static_folder, path)

# ------------------------------
# CSV normalization function
# ------------------------------
def detect_and_normalize_csv(df: pd.DataFrame):
    df.columns = [str(c).strip().replace(" ", "_").replace("\ufeff","") for c in df.columns]
    rename_map = {
        'Country': 'country',
        'Year': 'year',
        'Inflation_Rate': 'inflation',
        'Unemployment': 'unemployment',
        'Oil_Price': 'oil_price'
    }
    df = df.rename(columns=rename_map)
    required = {'country', 'year', 'inflation', 'unemployment', 'oil_price'}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must have columns: {required}. Found: {df.columns.tolist()}")

    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    df['inflation'] = pd.to_numeric(df['inflation'], errors='coerce')
    df['unemployment'] = pd.to_numeric(df['unemployment'], errors='coerce')
    df['oil_price'] = pd.to_numeric(df['oil_price'], errors='coerce')
    df = df.dropna(subset=['year', 'inflation', 'unemployment', 'oil_price'])
    df['year'] = df['year'].astype(int)
    return df

# ------------------------------
# Train models from CSV
# ------------------------------
def train_models_from_csv():
    global models, country_data
    trained = []
    skipped = {}

    try:
        df = pd.read_csv(DATA_FILE)
        tidy = detect_and_normalize_csv(df)

        for country, g in tidy.groupby("country"):
            g_sorted = g.sort_values("year")
            X = g_sorted[["year", "unemployment", "oil_price"]].values
            y = g_sorted["inflation"].values

            if len(X) < 2:
                skipped[country] = "Not enough data points"
                continue

            lr = LinearRegression()
            lr.fit(X, y)
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)

            models.setdefault(country, {})["linear"] = lr
            models[country]["randomforest"] = rf

            joblib.dump(lr, os.path.join(MODELS_DIR, f"{country}_linear.joblib"))
            joblib.dump(rf, os.path.join(MODELS_DIR, f"{country}_rf.joblib"))

            country_data[country] = {
                "years": g_sorted["year"].astype(int).tolist(),
                "inflation": g_sorted["inflation"].astype(float).tolist(),
                "unemployment": g_sorted["unemployment"].astype(float).tolist(),
                "oil_price": g_sorted["oil_price"].astype(float).tolist()
            }
            trained.append(country)

        print(f"✅ Training completed. Trained: {trained}, Skipped: {skipped}")
    except Exception as e:
        print(f"❌ Error reading/training CSV: {e}")

train_models_from_csv()

# ------------------------------
# Predict endpoint
# ------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    content = request.get_json(force=True)
    years = content.get("years", [])
    model_type = content.get("model", "linear")
    country = content.get("country")

    if not years:
        return jsonify({"error": "Please provide 'years' list"}), 400

    if not country:
        if len(models.keys()) == 1:
            country = list(models.keys())[0]
        else:
            return jsonify({"error": "Multiple countries available; please specify 'country'"}), 400

    if country not in models or model_type not in models[country]:
        return jsonify({"error": f"No trained model for country='{country}' and model='{model_type}'"}), 400

    model = models[country][model_type]
    hist_years = country_data[country]["years"]
    hist_inflation = country_data[country]["inflation"]
    last_unemployment = country_data[country]["unemployment"][-1]
    last_oil_price = country_data[country]["oil_price"][-1]

    preds = []
    for y in years:
        if y <= max(hist_years):
            X_new = [[y, last_unemployment, last_oil_price]]
            pred = model.predict(X_new)[0]
        else:
            lr_future = LinearRegression()
            X_hist = np.array(hist_years).reshape(-1, 1)
            y_hist = np.array(hist_inflation)
            lr_future.fit(X_hist, y_hist)
            pred = lr_future.predict(np.array([[y]]))[0]
        preds.append(round(float(pred), 2))

    return jsonify({
        "model": model_type,
        "country": country,
        "years": years,
        "inflation_predictions": preds
    })

# ------------------------------
# Remaining endpoints (unchanged)
# ------------------------------
@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "trained_countries": list(models.keys()),
        "has_models": bool(models),
        "historical_data": country_data
    })

@app.route("/top10", methods=["GET"])
def top10():
    year = request.args.get("year", type=int)
    latest_values = []

    for country, data in country_data.items():
        if year and year in data["years"]:
            idx = data["years"].index(year)
            latest_values.append({
                "country": country,
                "year": year,
                "unemployment": data["unemployment"][idx]
            })
        elif not year and data["years"]:
            latest_year = max(data["years"])
            idx = data["years"].index(latest_year)
            latest_values.append({
                "country": country,
                "year": latest_year,
                "unemployment": data["unemployment"][idx]
            })

    top10_countries = sorted(latest_values, key=lambda x: x["unemployment"], reverse=True)[:10]
    return jsonify(top10_countries)

@app.route("/all_countries", methods=["GET"])
def all_countries():
    year = request.args.get("year", type=int)
    filtered_values = []

    for country, data in country_data.items():
        if year and year in data["years"]:
            idx = data["years"].index(year)
            filtered_values.append({
                "country": country,
                "year": year,
                "unemployment": data["unemployment"][idx]
            })
        elif not year and data["years"]:
            latest_year = max(data["years"])
            idx = data["years"].index(latest_year)
            filtered_values.append({
                "country": country,
                "year": latest_year,
                "unemployment": data["unemployment"][idx]
            })

    all_countries_sorted = sorted(filtered_values, key=lambda x: x["unemployment"], reverse=True)
    return jsonify(all_countries_sorted)

@app.route("/oil_prices", methods=["GET"])
def all_oil_prices():
    oil_prices_set = set()
    for data in country_data.values():
        oil_prices_set.update(data["oil_price"])
    oil_prices_sorted = sorted(list(oil_prices_set))
    return jsonify(oil_prices_sorted)

@app.route("/years_oil", methods=["GET"])
def years_oil():
    years_set = set()
    for data in country_data.values():
        years_set.update(data["years"])
    years_sorted = sorted(list(years_set))
    return jsonify(years_sorted)

@app.route("/all_countries_oil", methods=["GET"])
def all_countries_oil():
    year = request.args.get("year", type=int)
    filtered_values = []

    for country, data in country_data.items():
        if year and year in data["years"]:
            idx = data["years"].index(year)
            filtered_values.append({
                "country": country,
                "year": year,
                "oil_price": data["oil_price"][idx]
            })
        elif not year and data["years"]:
            latest_year = max(data["years"])
            idx = data["years"].index(latest_year)
            filtered_values.append({
                "country": country,
                "year": latest_year,
                "oil_price": data["oil_price"][idx]
            })

    all_countries_sorted = sorted(filtered_values, key=lambda x: x["oil_price"], reverse=True)
    return jsonify(all_countries_sorted)

# ------------------------------
# Main
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
