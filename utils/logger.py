import os
import sys


LOG_FILE = os.getenv("LOG_FILE", "response_times_2.log")
PRINT_INFO = os.getenv("PRINT_INFO", True)
PRINT_ERROR = os.getenv("PRINT_ERROR", True)
PRINT_WARNING = os.getenv("PRINT_WARNING", True)


def log_response_time(event: str, response_time: float) -> None:
    with open(LOG_FILE, "a") as file:
        file.write(f"{event}: {response_time}s\n")


def print_info(message: str) -> None:
    if PRINT_INFO:
        GREEN = "\033[92m"
        RESET = "\033[0m"
        print(f"{GREEN}INFO:     {message}{RESET}", file=sys.stdout)


def print_error(message: str) -> None:
    if PRINT_ERROR:
        RED = "\033[91m"
        RESET = "\033[0m"
        print(f"{RED}ERROR:    {message}{RESET}", file=sys.stderr)


def print_warning(message: str) -> None:
    if PRINT_WARNING:
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        print(f"{YELLOW}WARNING:  {message}{RESET}", file=sys.stdout)
