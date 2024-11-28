import json

def output_results(config, output_results_path, env, state_dict):
    out_dict = {"env": env._to_dict(), "runs": state_dict}
    json_string = json.dumps(out_dict, separators=(",", ":"))
    with open(output_results_path, "w") as f:
        f.write(json_string)

    return out_dict
