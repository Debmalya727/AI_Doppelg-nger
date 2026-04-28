# run_local_login_server.py

from flask import Flask, redirect
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import json
import os

app = Flask(__name__)

CHROME_PATH = os.path.abspath("chrome-win64/chrome.exe")
CHROMEDRIVER_PATH = os.path.abspath("chromedriver-win64/chromedriver.exe")
COOKIES_OUTPUT_FILE = "instagram_cookies.json"

def get_driver():
    options = Options()
    options.binary_location = CHROME_PATH
    options.add_argument("--remote-debugging-port=9222")
    options.add_argument("--user-data-dir=selenium_session")  # Persistent session
    options.add_argument("--start-maximized")

    service = Service(CHROMEDRIVER_PATH)
    return webdriver.Chrome(service=service, options=options)

@app.route("/")
def home():
    return """
    <h1>Instagram Hybrid Login</h1>
    <a href="/start-login" target="_blank">👉 Click here to open Instagram Login</a><br>
    <p>Once you're logged in manually, close the browser tab, and this server will save your cookies.</p>
    """

@app.route("/start-login")
def start_login():
    driver = get_driver()
    driver.get("https://www.instagram.com/accounts/login/")

    print("🕒 Waiting for manual login...")
    time.sleep(120)  # Give the user time to log in and load feed

    print("✅ Extracting cookies...")
    cookies = driver.get_cookies()
    driver.quit()

    with open(COOKIES_OUTPUT_FILE, "w") as f:
        json.dump(cookies, f, indent=4)

    print(f"🍪 Cookies saved to {COOKIES_OUTPUT_FILE}")
    return redirect("/done")

@app.route("/done")
def done():
    return "<h2>✅ Login Completed. Cookies have been saved!</h2><p>You may now close this tab.</p>"

if __name__ == "__main__":
    print("🌐 Visit http://localhost:5000 to begin login...")
    app.run(port=5000)
