"""
TODO: Rework this into XLA init and SM Training ToolKit

This script currently does 2 things:
1. Setup XLA DT environment from SM environment
2. Allows xla_spawn.py script to take positional args which are not supported by SageMaker
"""


import argparse
import os
from pdb import run
import json
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.

    parser.add_argument("--training_script", type=str)
    parser.add_argument("--num_gpus", type=str, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--resource_config", type=str, default="/opt/ml/input/config/resourceconfig.json")
    parser.add_argument("--mesh_service_port", type=int, default=53957)
    parser.add_argument("--worker_port", type=int, default=43857)

    args, rem_args = parser.parse_known_args()

    with open(args.resource_config) as f:
        config = json.loads(f.read())
    current_host = config['current_host']
    rank = 0
    xrt_workers = []
    for i, host in enumerate(config['hosts']):
        if current_host == host:
            rank = i
        xrt_workers.append("localservice:%s;%s:%s" % (i, host, args.worker_port))
    os.environ["XRT_HOST_ORDINAL"] = str(rank)
    os.environ["XRT_SHARD_WORLD_SIZE"] = str(len(config['hosts']))
    os.environ["XRT_WORKERS"] = "|".join(xrt_workers)
    if len(config['hosts']) > 1:
        os.environ["XRT_MESH_SERVICE_ADDRESS"] = config['hosts'][0]+":"+str(args.mesh_service_port)


    os.environ["GPU_NUM_DEVICES"] = str(args.num_gpus)
    training_command = "python -m "
    training_command += (
        "torch_xla.distributed.xla_spawn "
    )
    training_command += f"--num_gpus {args.num_gpus} "
    training_command += args.training_script+" "

    for i in range(0, len(rem_args), 2):
        arg, value = rem_args[i], rem_args[i + 1]
        if value == "True":
            training_command += f"{arg} "
        elif value != "False":
            training_command += f"{arg} {value} "

    subprocess.check_call(training_command, shell=True)

