import sys
import os

# Add the /app directory to the Python path
sys.path.append("/app")


import threading
from flask import Flask, jsonify, request
import json
import psycopg2
from psycopg2 import OperationalError
import time
import hydra
from omegaconf import OmegaConf
from pathlib import Path
import custom.experiments.train as train_module
import custom.experiments.test as test_module
from hydra.core.global_hydra import GlobalHydra


app = Flask(__name__)


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


def run_test_job(overrides):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    hydra.initialize(config_path="../experiments/config", job_name="test_job")
    config = hydra.compose(config_name="config_test", overrides=overrides)
    test_module.main(config)


def run_train_job(overrides):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    hydra.initialize(config_path="../experiments/config", job_name="train_job")
    config = hydra.compose(config_name="config_train", overrides=overrides)
    train_module.main(config)


@app.route("/get-results", methods=["GET"])
def get_results():
    print("GET /get-results called")
    results_name = request.args.get("results_name")
    if not results_name:
        return jsonify({"error": "results_name parameter is required"}), 400

    results = read_job_data_from_db(results_name)

    if results is None:
        return (
            jsonify({"error": f"No data found for results_name: {results_name}"}),
            404,
        )

    return jsonify(results)


@app.route("/test", methods=["POST"])
def submit_test_job():
    job_data = request.get_json()
    print("POST /test called with data:", job_data)

    overrides = job_data.get("overrides", [])

    job_thread = threading.Thread(target=run_test_job, args=(overrides,))
    job_thread.start()

    return jsonify({"success": True, "estimated_time": 10})


@app.route("/train", methods=["POST"])
def submit_train_job():
    job_data = request.get_json()
    print("POST /train called with data:", job_data)

    overrides = job_data.get("overrides", [])

    job_thread = threading.Thread(target=run_train_job, args=(overrides,))
    job_thread.start()

    return jsonify({"success": True, "estimated_time": 10})


if __name__ == "__main__":
    port = int(os.getenv("FLASK_RUN_PORT", 5088))
    app.run(host="0.0.0.0", port=port)
