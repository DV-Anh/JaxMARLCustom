import json
from ..backend.handle_database import connect_db, write_job_data_to_db
from datetime import datetime


def output_results(config, output_results_path, env, state_dict):
    json_string = json.dumps(
        {"env": env._to_dict(), "runs": state_dict}, separators=(",", ":")
    )
    with open(output_results_path, "w") as f:
        f.write(json_string)

    if config.get("SAVE_TO_DB", False):
        connect_db()
        data_object = json.loads(json_string)
        job_name = config.get(
            "JOB_NAME", f'{config["ID"]}_data_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}_{config["ENV_NAME"]}_{config["algname"]}'
        )
        write_job_data_to_db(job_name, data_object)
