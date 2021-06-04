import zipfile
import os
from argparse import ArgumentParser
from shutil import copyfile
from misc.helpers import load_function
from lighter.config import Config
import torch
torch.multiprocessing.set_sharing_strategy('file_system')


def snapshot_code(path, fname, ignore=["data", "tmp", "runs"]):
    """Makes a snapshot of the code as a zip file to a target `fname` destination"""
    zipf = zipfile.ZipFile(fname, 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(path):
        # ignore temporary folders
        if any(i in root for i in ignore):
            continue
        for file in files:
            # only consider .py and .json files
            if file.endswith(".py") or file.endswith(".json"):
                zipf.write(os.path.join(root, file))
    zipf.close()


def parse_args():
    """Parses the arguments"""
    parser = ArgumentParser()
    parser.add_argument('--config', help='path to config file', type=str, required=True)
    parser.add_argument('--checkpoint', help='path of the checkpoint file to resume training', type=str)
    return parser.parse_args()


def run_experiments(config):
    """Loads the approach code and executes the experiments function"""
    approach = load_function(config.approach.module, 'experiments')
    approach(config)


def check_or_create_dirs():
    """Creates the basic temporary folder structure for the projects"""
    os.makedirs('tmp', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    os.makedirs('data', exist_ok=True)


def main():
    """Main entrance point for training"""
    check_or_create_dirs()
    options = parse_args()
    # # create config
    config = Config(path=options.config)
    # set resume checkpoint if set by args
    config.checkpoint = options.checkpoint if options.checkpoint else None
    # copy current config file to new experiment folder
    src = options.config
    dst = os.path.join(config.trainer.checkpoint_dir, config.trainer.experiment_name)
    if not os.path.exists(dst):
        os.makedirs(dst)
    copyfile(src, os.path.join(dst, 'config.json'))
    # copy code backup to experiments dir
    snapshot_code(os.getcwd(), os.path.join(dst, config.trainer.code_backup_filename))
    run_experiments(config)


if __name__ == "__main__":
    main()
