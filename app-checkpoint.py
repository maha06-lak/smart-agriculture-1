from flask import Flask, render_template, request, redirect, url_for, flash, session
import requests
import torch
import torch.nn.functional as F
import joblib
import numpy as np
import os
from torch_geometric.nn import GCNConv
import torch.nn as nn
from PIL import Image
import tensorflow as tf
from torchvision.models import vit_b_16
import sqlite3
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

app.secret_key = "secret123"

disease_solutions = {

"scab":[
"Remove infected leaves immediately",
"Apply fungicide spray weekly",
"Improve air circulation between plants",
"Avoid overhead watering",
"Use disease resistant crop varieties"
],

"rust":[
"Apply sulfur-based fungicide",
"Remove infected plant parts",
"Ensure proper plant spacing",
"Water plants early morning",
"Use rust resistant plant varieties"
]

}



soil_recommendations = {

"Alluvial soil":[
"Rice cultivation gives high yield",
"Wheat grows efficiently",
"Sugarcane farming suitable",
"Maize production recommended",
"Potato cultivation profitable",
"Vegetable farming ideal",
"Good irrigation improves productivity",
"Use organic compost for better soil health",
"Maintain moderate moisture",
"Crop rotation improves long term yield"
],

"Black Soil":[
"Cotton gives excellent yield",
"Soybean cultivation recommended",
"Groundnut farming suitable",
"Sunflower grows efficiently",
"Jowar cultivation possible",
"Maintain proper drainage",
"Use nitrogen fertilizers",
"Deep ploughing improves yield",
"Moderate irrigation required",
"Use crop rotation to avoid nutrient loss"
],

"Clay Soil":[
"Rice cultivation ideal",
"Lettuce grows well",
"Broccoli farming possible",
"Cabbage cultivation recommended",
"Beans suitable",
"Improve drainage system",
"Add organic matter regularly",
"Maintain balanced irrigation",
"Use raised bed farming",
"Mulching helps retain moisture"
],

"Red soil":[
"Millets grow efficiently",
"Groundnut cultivation recommended",
"Potato farming suitable",
"Pulses cultivation profitable",
"Oilseeds grow well",
"Add organic manure frequently",
"Use proper irrigation system",
"Crop rotation improves fertility",
"Use potassium fertilizers",
"Maintain moderate soil moisture"
]

}

def get_db():
    conn = sqlite3.connect('agri_data.db')
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE,
                        mobile TEXT,
                        email TEXT UNIQUE,
                        password TEXT)''')
    conn.commit()
    conn.close()

init_db()

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

svm = joblib.load("model/svm_model.pkl")
le_crop = joblib.load("model/le_crop.pkl")
le_soil = joblib.load("model/le_soil.pkl")
le_stage = joblib.load("model/le_stage.pkl")

gcn_model = GCN(6, 64)
gcn_model.load_state_dict(torch.load("model/gcn_model.pth", map_location='cpu'))
gcn_model.eval()

device = torch.device("cpu")

class_names = ["Rust & Scab","Healthy Leaf","Rust Disease","Scab Disease"]
model = load_model("model/my_model.h5")


pretrained_vit = vit_b_16(weights=None)

pretrained_vit.heads = nn.Linear(
    in_features=768,
    out_features=len(class_names)
)

pretrained_vit.load_state_dict(
    torch.load("model/model2_weights.pth", map_location=device)
)

pretrained_vit.to(device)
pretrained_vit.eval()




soil_model = tf.keras.models.load_model("model/soil_model1.h5")
soil_classes = ["Alluvial soil", "Black Soil", "Clay Soil", "Red soil"]



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/auth')
def auth():
    return render_template('auth.html')

@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    mobile = request.form['mobile']
    email = request.form['email']
    password = request.form['password']

    try:
        conn = get_db()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users (username, mobile, email, password) VALUES (?, ?, ?, ?)",
            (username, mobile, email, password)
        )
        conn.commit()
        conn.close()

        flash("Registration Successful")
        return redirect(url_for('auth'))

    except Exception as e:
        print(e)
        flash("Registration Failed")
        return redirect(url_for('auth'))
    
@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE username=?",
        (username,)
    ).fetchone()
    conn.close()

    if user and user["password"] == password:
        session['user_id'] = user['id']
        session['username'] = user['username']
        return redirect(url_for('dashboard'))

    flash("Invalid Login")
    return redirect(url_for('auth'))

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('auth'))
    
    return render_template('dashboard.html', username=session['username'])



API_KEY = "86d8242662aabd63112ea6bc8c6570cb"

def get_weather(city):
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    
    response = requests.get(url)
    data = response.json()
    
    if response.status_code == 200:
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        return temp, humidity
    else:
        return None, None

@app.route('/irrigation', methods=["GET", "POST"])
def irrigation():

    prediction = None
    message = None
    temp = None
    humidity = None
    tips = []

    crops = le_crop.classes_
    soils = le_soil.classes_
    stages = le_stage.classes_

    if request.method == "POST":

        crop_name = request.form["crop"]
        soil = request.form["soil"]
        stage = request.form["stage"]
        moi = float(request.form["moi"])
        city = request.form["city"]

        temp, humidity = get_weather(city)

        if temp is None:
            return "Weather API Error"

        crop_encoded = le_crop.transform([crop_name])[0]
        soil_encoded = le_soil.transform([soil])[0]
        stage_encoded = le_stage.transform([stage])[0]

        X = torch.tensor([[crop_encoded, soil_encoded, stage_encoded, moi, temp, humidity]], dtype=torch.float)
        edge_index = torch.tensor([[0],[0]], dtype=torch.long)

        with torch.no_grad():
            embedding = gcn_model(X, edge_index)

        embedding = embedding.numpy()
        prediction = int(svm.predict(embedding)[0])


        if prediction == 1:

            # CARROT
            #print(crop_name)
            if crop_name == "Carrot":
                if moi < 0.35:
                    message = "Irrigation required for Carrot crop"
                    tips = [
                        "Maintain consistent soil moisture for proper root growth.",
                        "Avoid dry soil conditions which cause root cracking.",
                        "Use light and frequent irrigation."
                    ]
                else:
                    message = "Moisture level sufficient for Carrot"
                    tips = [
                        "No immediate irrigation required.",
                        "Monitor soil moisture regularly."
                    ]

            # POTATO
            elif crop_name == "Potato":
                if moi < 0.40:
                    message = "Irrigation recommended for Potato"
                    tips = [
                        "Potatoes need adequate moisture during tuber formation.",
                        "Avoid water stress during early growth stage.",
                        "Apply moderate irrigation."
                    ]
                else:
                    message = "Soil moisture adequate for Potato"
                    tips = [
                        "Delay irrigation for now.",
                        "Check soil moisture again in 1–2 days."
                    ]

            # TOMATO
            elif crop_name == "Tomato":
                if moi < 0.45:
                    message = "Irrigation required for Tomato plants"
                    tips = [
                        "Tomatoes require regular watering for fruit development.",
                        "Use drip irrigation to maintain consistent moisture.",
                        "Avoid wetting leaves to reduce disease risk."
                    ]
                else:
                    message = "Moisture level sufficient for Tomato"
                    tips = [
                        "No irrigation required at the moment.",
                        "Continue monitoring soil moisture."
                    ]

            # CHILLI
            elif crop_name == "Chilli":
                if moi < 0.40:
                    message = "Irrigation recommended for Chilli crop"
                    tips = [
                        "Maintain moderate soil moisture.",
                        "Avoid excessive irrigation which can damage roots.",
                        "Water plants early morning or evening."
                    ]
                else:
                    message = "Soil moisture adequate for Chilli"
                    tips = [
                        "Irrigation can be delayed.",
                        "Monitor moisture levels during hot weather."
                    ]

            # WHEAT
            elif crop_name == "Wheat":
                if moi < 0.35:
                    message = "Irrigation required for Wheat"
                    tips = [
                        "Provide irrigation during critical growth stages.",
                        "Avoid water stagnation in the field.",
                        "Light irrigation is usually sufficient."
                    ]
                else:
                    message = "Moisture level sufficient for Wheat"
                    tips = [
                        "Irrigation not required immediately.",
                        "Check soil moisture after 1–2 days."
                    ]
            if temp > 35:
                tips.append("High temperature detected — consider increasing irrigation frequency.")

            if humidity < 40:
                tips.append("Low humidity may increase soil moisture loss.")

        else:

            message = "No Irrigation Required Today"

            tips = [
                "Today irrigation is not required.",
                "Check soil moisture again tomorrow before watering.",
                "Monitor plant leaves and soil condition regularly."
            ]

    return render_template(
        "irrigation.html",
        prediction=prediction,
        message=message,
        tips=tips,
        temp=temp,
        humidity=humidity,
        crops=crops,
        soils=soils,
        stages=stages
    )

@app.route('/disease', methods=["GET", "POST"])
def disease():

    predictions = []
    image_path = None

    if request.method == "POST":

        file = request.files["image"]

        if file and file.filename != "":

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224,224))
            img_array = image.img_to_array(img)

            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            prediction = model.predict(img_array)

            predicted_index = np.argmax(prediction)
            disease = class_names[predicted_index]

            confidence = float(np.max(prediction)) * 100


            if "Scab" in disease and "Rust" in disease:

                predictions.append({
                    "disease": "Scab",
                    "confidence": confidence,
                    "recommendations": disease_solutions.get("scab",[])
                })

                predictions.append({
                    "disease": "Rust",
                    "confidence": confidence,
                    "recommendations": disease_solutions.get("rust",[])
                })

            elif "Scab" in disease:

                predictions.append({
                    "disease": "Scab",
                    "confidence": confidence,
                    "recommendations": disease_solutions.get("scab",[])
                })

            elif "Rust" in disease:

                predictions.append({
                    "disease": "Rust",
                    "confidence": confidence,
                    "recommendations": disease_solutions.get("rust",[])
                })
            elif "Healthy Leaf" in disease:
                 predictions.append({
                    "disease": "Healthy Leaf",
                    "confidence": confidence,
                    "recommendations": " "
                })

    return render_template(
        "disease.html",
        predictions=predictions,
        image_path=image_path
    )

@app.route('/soil', methods=["GET", "POST"])
def soil():
    result = None
    recommendations = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        image_path = filepath

        img = tf.keras.preprocessing.image.load_img(
            filepath,
            target_size=(224, 224)
        )

        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        img_array = img_array / 255.0

        prediction = soil_model.predict(img_array)

        result = soil_classes[np.argmax(prediction)]

        recommendations = soil_recommendations.get(result, [])

    return render_template("soil.html",
                           result=result,
                           recommendations=recommendations,
                           image_path=image_path)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=False,port=2345)
