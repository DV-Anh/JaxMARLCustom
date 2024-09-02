import psycopg2
from psycopg2 import OperationalError
import time
import json


def connect_db():
    retries = 5
    while retries > 0:
        try:
            conn = psycopg2.connect(
                dbname="gamedb", user="user", password="password", host="database"
            )
            return conn
        except OperationalError as e:
            print(f"Database connection failed: {str(e)}")
            retries -= 1
            time.sleep(5)
    raise Exception("Failed to connect to the database after multiple attempts")


def read_job_data_from_db(name):
    conn = connect_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT data FROM job_results WHERE name = %s", (name,))
        result = cursor.fetchone()
        if result:
            print("Read database successfully")
            return result[0]
        else:
            print(f"No data found for name: {name}")
            return None
    finally:
        cursor.close()
        conn.close()


def write_job_data_to_db(name, data):
    conn = connect_db()
    cursor = conn.cursor()
    try:
        json_data = json.dumps(data)

        cursor.execute(
            "INSERT INTO job_results (name, data) VALUES (%s, %s) ON CONFLICT (name) DO NOTHING",
            (name, json_data),
        )
        conn.commit()
        print("Data written to the database successfully")
    except Exception as e:
        print(f"Failed to write data to the database: {str(e)}")
    finally:
        cursor.close()
        conn.close()
