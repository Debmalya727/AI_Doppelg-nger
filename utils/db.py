# utils/db.py
import mysql.connector

# --- IMPORTANT: UPDATE THESE DETAILS ---
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '2004',  # <-- Put your MySQL password here
    'database': 'doppelganger_db'
}
# --------------------------------------

def get_db_connection():
    """Establishes a connection to the MySQL database."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except mysql.connector.Error as err:
        print(f"[ERROR] MySQL Connection Failed: {err}")
        return None