# Adapted from https://github.com/monniert/dti-clustering/blob/b57a77d4c248b16b4b15d6509b6ec493c53257ef/src/utils/logger.py
import logging
import time
import socket
from datetime import datetime
import os
import shutil
from lib.utils import coerce_to_path_and_check_exist


class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_info(s):
    print(TerminalColors.OKBLUE + "[" + get_time() + "] " + str(s) + TerminalColors.ENDC)


def print_warning(s):
    print(TerminalColors.WARNING + "[" + get_time() + "] WARN " + str(s) + TerminalColors.ENDC)


def print_error(s):
    print(TerminalColors.FAIL + "[" + get_time() + "] ERROR " + str(s) + TerminalColors.ENDC)


def get_logger(log_dir, name):
    log_dir = coerce_to_path_and_check_exist(log_dir)
    logger = logging.getLogger(name)
    file_path = log_dir / "{}.log".format(name)
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


def print_info(s):
    print(TerminalColors.OKBLUE + "[" + get_time() + "] " + str(s) + TerminalColors.ENDC)


def print_and_log_info(logger, string):
    logger.info(string)


def ensure_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def init_logger(save_path, trainer_dir, trainer_logger_name=None):
    if not os.path.exists(trainer_dir):
        os.makedirs(trainer_dir)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if trainer_logger_name is not None:
        trainer_logger = get_logger(trainer_dir, trainer_logger_name)
    else:
        trainer_logger = get_logger(trainer_dir, "trainer")
    print_and_log_info(trainer_logger, "Model's weights at {}".format(save_path))
    return trainer_logger

