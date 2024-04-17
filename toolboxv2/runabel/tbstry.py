import os
import threading
import time

from PIL import Image

from toolboxv2 import App, show_console, get_app, Spinner

path_to_installer = r"C:\Users\Markin\Workspace\tb_ui_installer\dist\UiInstallerTB\UiInstallerTB.exe"


def create_image():
    # Generate an image and draw a pattern
    image = Image.open('./web/webapp/icon-512.png')
    return image


NAME = 'TBtray'

show = True


def showC(icon, item):
    global show
    if not show_console(True):
        icon.notify('Console is already open!')
        time.sleep(4)
        icon.remove_notification()
    else:
        show = False


def hideC(icon, item):
    global show
    if not show_console(False):
        icon.notify('Console is closed!')
        time.sleep(6)
        icon.remove_notification()
    else:
        show = True


def open_ui(*args, **kwargs):
    from toolboxv2.mods.CloudM.UI.vue import MyApp
    MyApp().mainloop()


def exit_(icon, item):
    icon.stop()
    app = get_app(from_=f"{NAME}.exit")
    app.exit_main()
    app.exit()


def showI(icon, item):
    if os.system(f"start {path_to_installer}") != 0:
        icon.notify('Error finding!')
        time.sleep(6)
        icon.remove_notification()


def get_initial_icon(app: App):
    import pystray
    icon = pystray.Icon(name="RE-Simple-ToolBoxV2",
                        title=f"ToolBox {app.version}",
                        icon=create_image(),
                        menu=pystray.Menu(
                            pystray.MenuItem("Open UI", open_ui),
                            pystray.MenuItem("Open Installer UI", showI),
                            pystray.MenuItem("Show Console", showC, checked=lambda item: show),
                            pystray.MenuItem("Hide Console", hideC, checked=lambda item: not show),
                            pystray.MenuItem("Exit", exit_),
                        ))
    return icon


async def run(app, args):
    with Spinner("Initial"):
        icon = get_initial_icon(app)

    threading.Thread(target=icon.run, daemon=True).start()
    await app.run_runnable('cli')
