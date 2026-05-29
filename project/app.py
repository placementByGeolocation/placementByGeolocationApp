from flask import Flask, render_template, request, jsonify
import random

app = Flask(__name__)

# ---------------------------------------------------
# MAIN PAGE
# ---------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

# ---------------------------------------------------
# ANALYZE POINT
# ---------------------------------------------------

@app.route("/analyze", methods=["POST"])
def analyze():

    data = request.get_json()

    lat = data.get("lat")
    lng = data.get("lng")

    # ---------------------------------------------------
    # ТУТ БУДЕТ ML / API
    # ---------------------------------------------------

    decision = random.choice([True, False])

    reason = (
        "высокий пешеходный трафик"
        if decision
        else "слишком много конкурентов рядом"
    )

    return jsonify({
        "success": decision,
        "reason": reason,
        "lat": lat,
        "lng": lng
    })

# ---------------------------------------------------
# RUN
# ---------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)