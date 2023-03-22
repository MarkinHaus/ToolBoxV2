import datetime

import logging
import os
from logging.handlers import SocketHandler
from math import inf as infinity

from toolboxv2.Style import Style, remove_styles

loggerNameOfToolboxv2 = 'toolboxV2'


def setup_logging(level: int, name=loggerNameOfToolboxv2, online_level=None, is_online=False, file_level=None,
                  interminal=False):
    global loggerNameOfToolboxv2

    if not online_level:
        online_level = level

    if not file_level:
        file_level = level

    if not os.path.exists(f"./logs"):
        os.makedirs(f"./logs")
        open(f"logs/Logs.info", "a").close()

    loggerNameOfToolboxv2 = name

    if level not in [logging.CRITICAL, logging.FATAL, logging.ERROR, logging.WARNING, logging.WARN, logging.INFO,
                     logging.DEBUG, logging.NOTSET]:
        raise ValueError(
            f"level must be on of (CRITICAL, FATAL, ERROR, WARNING, WARN, INFO, DEBUG, NOTSET)"
            f" logging level is : {level}")

    if online_level not in [logging.CRITICAL, logging.FATAL, logging.ERROR, logging.WARNING, logging.WARN, logging.INFO,
                            logging.DEBUG, logging.NOTSET]:
        raise ValueError(
            f"online_level must be on of (CRITICAL, FATAL, ERROR, WARNING, WARN, INFO, DEBUG, NOTSET)"
            f" logging level is : {online_level}")

    if file_level not in [logging.CRITICAL, logging.FATAL, logging.ERROR, logging.WARNING, logging.WARN, logging.INFO,
                          logging.DEBUG, logging.NOTSET]:
        raise ValueError(
            f"file_level must be on of (CRITICAL, FATAL, ERROR, WARNING, WARN, INFO, DEBUG, NOTSET)"
            f" logging level is : {file_level}")

    filename = f"Logs-{name}-{datetime.datetime.today().strftime('%Y-%m-%d')}-" \
               f"{['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET', ][[50, 40, 30, 20, 10, 0].index(level)]}"

    log_info_data = {
        filename: 0,
        "H": "localhost",
        "P": 62435
    }

    with open(f"logs/Logs.info", "r") as li:
        log_info_data_str = li.read()
        try:
            log_info_data = eval(log_info_data_str)
        except SyntaxError:
            if len(log_info_data_str):
                print(Style.RED(Style.Bold("Could not parse log")))

        if filename not in log_info_data.keys():
            log_info_data[filename] = 0

        if not os.path.exists(f"logs/{filename}.log"):
            log_info_data[filename] = 0
            print("new log file")

        if os.path.exists(f"logs/{filename}.log"):
            log_info_data[filename] += 1

            while os.path.exists(f"logs/{filename}#{log_info_data[filename]}.log"):
                log_info_data[filename] += 1

            try:
                os.rename(f"logs/{filename}.log",
                          f"logs/{filename}#{log_info_data[filename]}.log")
            except PermissionError:
                print(Style.RED(Style.Bold(f"Could not rename log file appending on {filename}")))

    with open(f"logs/Logs.info", "w") as li:
        if len(log_info_data.keys()) >= 7:
            log_info_data = {
                filename: log_info_data[filename],
                "H": log_info_data["H"],
                "P": log_info_data["P"]
            }

        li.write(str(log_info_data))

    with open(f"logs/{filename}.log", "a"):
        pass

    if interminal:
        logging.basicConfig(level=level, format=f"%(asctime)s %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    else:
        logging.basicConfig(level=level, filename=f"logs/{filename}.log",
                            format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                            datefmt="%Y-%m-%d %H:%M:%S"
                            )

    logger = logging.getLogger(name)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s:%(lineno)d - '
                                  '%(message)s')

    if interminal:
        handler = logging.FileHandler(f"logs/{filename}.log")
        handler.setFormatter(formatter)
        handler.setLevel(file_level)
        logger.addHandler(handler)

    if is_online:
        handler = SocketHandler(log_info_data["H"], log_info_data["P"])
        handler.setFormatter(formatter)
        handler.setLevel(online_level)
        logger.addHandler(handler)

    logger.setLevel(level)

    return logger


def get_logger() -> logging.Logger:
    logger = logging.getLogger(loggerNameOfToolboxv2)
    return logger


def unstyle_log_files(filename):
    text = ""
    with open(filename, "r") as f:
        text = f.read()

    text = remove_styles(text)
    text += "\n no-styles \n"

    with open(filename, "w") as f:
        f.write(text)


def edit_log_files(name: str, date: str, level: int, n=1, m=infinity, do=os.remove):
    year, month, day = date.split('-')
    if day.lower() == "xx":
        for i in range(1, 32):
            n_date = year + '-' + month + '-' + ('0' if i < 10 else '') + str(i)
            _edit_many_log_files(name, n_date, level, n, m, do)
    else:
        _edit_many_log_files(name, date, level, n, m, do)


def _edit_many_log_files(name, date, level, n, m, d):
    filename = f"Logs-{name}-{date}-" \
               f"{['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET', ][[50, 40, 30, 20, 10, 0].index(level)]}"
    if not n and os.path.exists(f"logs/{filename}.log"):
        print(f"editing {filename}.log")
        d(f"logs/{filename}.log")

    if not n:
        n += 1
    while os.path.exists(f"logs/{filename}#{n}.log"):
        if n >= m:
            break
        print(f"editing {filename}#{n}.log")
        d(f"logs/{filename}#{n}.log")
        n += 1


#edit_log_files("toolbox-test", '2023-02-XX', logging.NOTSET, 0, do=unstyle_log_files)
