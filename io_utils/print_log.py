RESET   = "\033[0m"
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
GRAY    = "\033[90m"

def log(tag, msg):
    colors = {
        "CONSUMER" : BLUE,
        "PROCESS"  : YELLOW,
        "DONE"     : GREEN,
        "ERROR"    : RED,
        "STOP"     : MAGENTA,
        "DEBUG"    : GRAY
    }
    color = colors.get(tag, RESET)
    print(f"{color}[{tag.center(10)}]{RESET} {msg}")