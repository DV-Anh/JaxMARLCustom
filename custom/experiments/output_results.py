import json
from ..backend.app import connect_db, write_job_data_to_db


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
            "JOB_NAME", f'data_{config["ENV_NAME"]}_{config["algname"]}'
        )
        write_job_data_to_db(job_name, data_object)
