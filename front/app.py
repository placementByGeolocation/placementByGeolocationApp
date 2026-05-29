from flask import Flask, render_template, request, jsonify
import requests

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

    try:

        data = request.get_json()

        lat = data["lat"]
        lng = data["lng"]

        # ---------------------------------------------------
        # ВЫЗОВ FASTAPI
        # ---------------------------------------------------

        response = requests.post(

            "http://127.0.0.1:8000/forward/",

            json={
                "lat": lat,
                "lon": lng
            },

            timeout=10
        )

        # ---------------------------------------------------
        # ПРОВЕРКА ОШИБКИ API
        # ---------------------------------------------------

        response.raise_for_status()

        result = response.json()

        # ---------------------------------------------------
        # ВОЗВРАЩАЕМ FRONTEND
        # ---------------------------------------------------

        return jsonify({
            "success": True,
            "prediction": result
        })

    except requests.exceptions.RequestException as e:

        return jsonify({
            "success": False,
            "error": f"API request failed: {str(e)}"
        }), 500

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# ---------------------------------------------------
# RUN
# ---------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)