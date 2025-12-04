from apps import app  # re-export Flask app for convenience

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
