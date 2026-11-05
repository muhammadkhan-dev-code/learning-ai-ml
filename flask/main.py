
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Welcome to Flask Application</h1>"

# Registration page
@app.route("/form")
def form():
    return render_template("index.html")

# Login page
@app.route("/login")
def login():
    return render_template("login.html")

if __name__ == "__main__":
    app.run(debug=True)
