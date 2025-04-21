import os
import sqlite3
from datetime import datetime
import pytest

DB_PATH = "monitor/logs/logs.db"

# Set up the database and table for testing
def setup_module(module):
    os.makedirs("monitor/logs", exist_ok=True)
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
            response_time REAL,
            api_type TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Test case for inserting and retrieving a feedback log.
def test_log_feedback():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute('''
        INSERT INTO logs (timestamp, event, query, response, feedback, response_time, api_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (now, 'feedback', 'Do I pay VAT?', 'Yes, depending on your tax regime.', '👍', None, None))
    conn.commit()
    cursor.execute("SELECT * FROM logs WHERE event='feedback' AND feedback='👍'")
    result = cursor.fetchone()
    conn.close()
    assert result is not None
    assert result[2] == 'feedback'
    assert result[5] == '👍'

# Test case for inserting and verifying an 'answered' log.
def test_log_answered():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute('''
        INSERT INTO logs (timestamp, event, query, response, feedback, response_time, api_type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (now, 'answered', 'How do I open a Partita IVA?', 'Go to Agenzia delle Entrate...', None, 1.4, None))
    conn.commit()
    cursor.execute("SELECT * FROM logs WHERE event='answered'")
    result = cursor.fetchone()
    conn.close()
    assert result is not None
    assert result[2] == 'answered'
