import ctypes

TBRUNNER_console_viabel = True


def show_console(show=True):
    global TBRUNNER_console_viabel
    """Brings up the Console Window."""
    try:
        if show and not TBRUNNER_console_viabel:
            # Show console
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 4)
            TBRUNNER_console_viabel = True
        elif not show and TBRUNNER_console_viabel:
            # Hide console
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)
            TBRUNNER_console_viabel = False
    except:
        print(f"Could not show_console {show=}", )
