from datetime import datetime
from pathlib import Path
import json
from pathlib import Path


def make_run_id():
    run_id = datetime.now().strftime(r"%m%d_%H%M%S")
    return run_id

def make_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f'Created folder {path}')

def save_args_to_json(args, path):
    args_dict = args.__dict__
    file_name = path + "/args.json"
    write_json(args_dict, file_name)
    print(f'Created file {file_name}')

def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)



