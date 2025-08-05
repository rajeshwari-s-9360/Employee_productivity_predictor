from flask import Flask, render_template, request
from waitress import serve
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import webbrowser
import threading

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
os.makedirs("static/plots", exist_ok=True)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        try:
            features = [
                float(request.form['targeted_productivity']),
                float(request.form['smv']),
                float(request.form['wip']),
                float(request.form['over_time']),
                float(request.form['incentive']),
                float(request.form['idle_time']),
                float(request.form['idle_men']),
                int(request.form['no_of_style_change']),
                int(request.form['no_of_workers'])
            ]
            prediction = model.predict([features])[0]

            # Create and save plots
            x = np.random.rand(10)
            y = np.random.rand(10)

            plt.figure(); plt.bar(range(10), np.random.randint(1, 100, 10))
            plt.title("Bar Chart"); plt.savefig("static/plots/bar.png"); plt.close()

            plt.figure(); plt.scatter(x, y); plt.title("Scatter Plot")
            plt.savefig("static/plots/scatter.png"); plt.close()

            plt.figure(); plt.plot(x, y); plt.title("Line Plot")
            plt.savefig("static/plots/line.png"); plt.close()

            plt.figure(); plt.pie([30, 20, 50], labels=['A', 'B', 'C'], autopct='%1.1f%%')
            plt.title("Pie Chart"); plt.savefig("static/plots/pie.png"); plt.close()

            plt.figure(); plt.hist(np.random.randn(100), bins=10)
            plt.title("Histogram"); plt.savefig("static/plots/hist.png"); plt.close()

            plt.figure(); plt.boxplot(np.random.randn(100))
            plt.title("Boxplot"); plt.savefig("static/plots/box.png"); plt.close()

            return render_template("submit.html", prediction=prediction)
        except Exception as e:
            return f"❌ Error: {e}"
    return render_template("predict.html")

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8000")

if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    print("✅ Running on http://127.0.0.1:8000")
    serve(app, host="0.0.0.0", port=8000)
