# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
TODO: Move this functionality to SM Training ToolKit

This script launches a single training process per GPU.
It requires a positional argument which is unsupported by SageMaker.

A simple launcher script for GPU training

Inspired by https://github.com/huggingface/transformers/blob/master/examples/legacy/seq2seq/xla_spawn.py

::
    >>> python xla_spawn.py --num_gpus=NUM_CORES_YOU_HAVE
               YOUR_TRAINING_SCRIPT.py (--arg1 --arg2 --arg3 and all other
               arguments of your training script)

"""


import importlib
import sys
from argparse import REMAINDER, ArgumentParser
from pathlib import Path
  
import torch_xla.distributed.xla_multiprocessing as xmp


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = ArgumentParser(
        description=(
            "PyTorch GPU distributed training launch "
            "helper utility that will spawn up "
            "multiple distributed processes"
        )
    )

    # Optional arguments for the launch helper
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs cores to use.")

    # positional
    parser.add_argument(
        "training_script",
        type=str,
        help=(
            "The full path to the single GPUs training "
            "program/script to be launched in parallel, "
            "followed by all the arguments for the "
            "training script"
        ),
    )

    # rest from the training program
    parser.add_argument("training_script_args", nargs=REMAINDER)

    return parser.parse_args()


def main():
    args = parse_args()

    # Import training_script as a module.
    script_fpath = Path(args.training_script)
    sys.path.append(str(script_fpath.parent.resolve()))
    mod_name = script_fpath.stem
    mod = importlib.import_module(mod_name)

    # Patch sys.argv
    sys.argv = [args.training_script] + args.training_script_args

    xmp.spawn(mod._mp_fn, args=(), nprocs=args.num_gpus)


if __name__ == "__main__":
    main()
