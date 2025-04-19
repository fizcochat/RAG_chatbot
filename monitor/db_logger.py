import sqlite3
from datetime import datetime
import pandas as pd
import os

DB_DIR = "monitor/logs"
DB_PATH = os.path.join(DB_DIR, "logs.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)  # ensure folder exists
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                event TEXT,
                query TEXT,
                response TEXT,
                feedback TEXT,
                response_time REAL
            )
        ''')
        conn.commit()
        conn.close()


def log_event(event, query=None, response=None, feedback=None, response_time=None):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Safely cast response_time to float if not None
    if response_time is not None:
        response_time = float(response_time)

    cursor.execute('''
        INSERT INTO logs (timestamp, event, query, response, feedback, response_time)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        str(event),
        str(query) if query else None,
        str(response) if response else None,
        str(feedback) if feedback else None,
        response_time
    ))
    conn.commit()
    conn.close()


def get_all_logs():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, timestamp, event, query, response, feedback, response_time FROM logs')
    rows = cursor.fetchall()
    conn.close()
    return rows

def export_logs_to_csv(db_path="monitor/logs/logs.db", csv_path="monitor/logs/logs_export.csv"):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM logs", conn)
    df.to_csv(csv_path, index=False)
    conn.close()
