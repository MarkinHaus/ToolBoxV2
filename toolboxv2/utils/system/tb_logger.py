import datetime
import logging
import os
from logging.handlers import SocketHandler

from ..extras.Style import Style, remove_styles

loggerNameOfToolboxv2 = 'toolboxV2'


def setup_logging(level: int, name=loggerNameOfToolboxv2, online_level=None, is_online=False, file_level=None,
                  interminal=False, logs_directory="../logs", app_name="main"):
    global loggerNameOfToolboxv2

    if not online_level:
        online_level = level

    if not file_level:
        file_level = level

    if not os.path.exists(logs_directory):
        os.makedirs(logs_directory, exist_ok=True)
    if not os.path.exists(logs_directory + "/Logs.info"):
        open(f"{logs_directory}/Logs.info", "a").close()

    loggerNameOfToolboxv2 = name

    available_log_levels = [logging.CRITICAL, logging.FATAL, logging.ERROR, logging.WARNING, logging.WARN, logging.INFO,
                            logging.DEBUG, logging.NOTSET]

    if level not in available_log_levels:
        raise ValueError(f"level must be one of {available_log_levels}, but logging level is {level}")

    if online_level not in available_log_levels:
        raise ValueError(f"online_level must be one of {available_log_levels}, but logging level is {online_level}")

    if file_level not in available_log_levels:
        raise ValueError(f"file_level must be one of {available_log_levels}, but logging level is {file_level}")

    log_date = datetime.datetime.today().strftime('%Y-%m-%d')
    log_levels = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]
    log_level_index = log_levels.index(logging.getLevelName(level))

    filename = f"Logs-{name}-{log_date}-{log_levels[log_level_index]}"
    log_filename = f"{logs_directory}/{filename}.log"

    log_info_data = {
        filename: 0,
        "H": "localhost",
        "P": 62435
    }

    with open(f"{logs_directory}/Logs.info") as li:
        log_info_data_str = li.read()
        try:
            log_info_data = eval(log_info_data_str)
        except SyntaxError:
            if log_info_data_str:
                print(Style.RED(Style.Bold("Could not parse log info data")))

        if filename not in log_info_data:
            log_info_data[filename] = 0

        if not os.path.exists(log_filename):
            log_info_data[filename] = 0
            print("new log file")

        if os.path.exists(log_filename):
            log_info_data[filename] += 1

            while os.path.exists(f"{logs_directory}/{filename}#{log_info_data[filename]}.log"):
                log_info_data[filename] += 1

            try:
                os.rename(log_filename,
                          f"{logs_directory}/{filename}#{log_info_data[filename]}.log")
            except PermissionError:
                pass
                # print(Style.YELLOW(Style.Bold(f"Could not rename log file appending on {filename}")))

    with open(f"{logs_directory}/Logs.info", "w") as li:
        if len(log_info_data.keys()) >= 7:
            log_info_data = {
                filename: log_info_data[filename],
                "H": log_info_data["H"],
                "P": log_info_data["P"]
            }
        li.write(str(log_info_data))

    try:
        with open(log_filename, "a"):
            pass
    except OSError:
        log_filename = f"{logs_directory}/Logs-Test-{log_date}-{log_levels[log_level_index]}.log"
        with open(log_filename, "a"):
            pass

    logger = logging.getLogger(name)

    logger.setLevel(level)
    # Prevent logger from propagating to parent loggers
    logger.propagate = False

    terminal_format = f"{app_name} %(asctime)s %(levelname)s %(name)s - %(message)s"
    file_format = f"{app_name} %(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(funcName)s:%(lineno)d - %(message)s"

    # Configure handlers
    handlers = []

    # File handler (always added)
    file_handler = logging.FileHandler(log_filename)
    file_handler.setFormatter(logging.Formatter(file_format))
    file_handler.setLevel(file_level)
    handlers.append(file_handler)

    # Terminal handler (if requested)
    if interminal:
        terminal_handler = logging.StreamHandler()
        terminal_handler.setFormatter(logging.Formatter(terminal_format))
        terminal_handler.setLevel(level)
        handlers.append(terminal_handler)

    # Socket handler (if requested)
    if is_online:
        socket_handler = SocketHandler(log_info_data["H"], log_info_data["P"])
        socket_handler.setFormatter(logging.Formatter(file_format))
        socket_handler.setLevel(online_level)
        handlers.append(socket_handler)

    # Add all handlers to logger
    for handler in handlers:
        logger.addHandler(handler)

    return logger, filename


def get_logger() -> logging.Logger:
    return logging.getLogger(loggerNameOfToolboxv2)


def unstyle_log_files(filename):
    text = ""
    with open(filename) as f:
        text = f.read()
    text = remove_styles(text)
    text += "\n no-styles \n"
    with open(filename, "w") as f:
        f.write(text)


def edit_log_files(name: str, date: str, level: int, n=1, m=float('inf'), do=os.remove):
    year, month, day = date.split('-')
    if day.lower() == "xx":
        for i in range(1, 32):
            n_date = year + '-' + month + '-' + ('0' if i < 10 else '') + str(i)
            _edit_many_log_files(name, n_date, level, n, m, do)
    else:
        _edit_many_log_files(name, date, level, n, m, do)


def _edit_many_log_files(name, date, level, log_file_number, max_number, do):
    log_levels = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"]
    log_level_index = log_levels.index(logging.getLevelName(level))
    filename = f"Logs-{name}-{date}-{log_levels[log_level_index]}"
    if not log_file_number and os.path.exists(f"logs/{filename}.log"):
        print(f"editing {filename}.log")
        do(f"logs/{filename}.log")
    if not log_file_number:
        log_file_number += 1
    while os.path.exists(f"logs/{filename}#{log_file_number}.log"):
        if log_file_number >= max_number:
            break
        print(f"editing {filename}#{log_file_number}.log")
        do(f"logs/{filename}#{log_file_number}.log")
        log_file_number += 1

# edit_log_files("toolbox-test", '2023-02-XX', logging.NOTSET, 0, do=unstyle_log_files)
