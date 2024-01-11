import json
import os
import platform

from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from toolboxv2 import get_logger, App, get_app
from toolboxv2.utils.state_system import TbState, get_state_from_app

router = APIRouter(
    prefix="/api",
)


def save_mod_snapshot(app, mod_name, provider=None, tb_state: TbState or None = None):
    if app is None:
        app = get_app(from_="Api.start.installer")
    if provider is None:
        provider = app.config_fh.get_file_handler("provider::")
    if provider is None:
        raise ValueError("No provider specified")
    if tb_state is None:
        tb_state: TbState = get_state_from_app(app, simple_core_hub_url=provider)
    mod_data = tb_state.mods.get(mod_name)
    if mod_data is None:
        mod_data = tb_state.mods.get(mod_name + ".py")

    if mod_data is None:
        app.print(f"Valid ar : {list(tb_state.mods.keys())}")
        return None

    if not os.path.exists("./installer"):
        os.mkdir("./installer")

    json_data = {"Name": mod_name,
                 "mods": [mod_data.url],
                 "runnable": None,
                 "requirements": None,
                 "additional-dirs": None,
                 mod_name: {
                     "version": mod_data.version,
                     "shasum": mod_data.shasum,
                     "provider": mod_data.provider,
                     "url": mod_data.url
                 }}
    installer_path = f"./installer/{mod_name}-installer.json"
    if os.path.exists(installer_path):
        with open(installer_path, "r") as installer_file:
            file_data: dict = json.loads(installer_file.read())
            if len(file_data.get('mods', [])) > 1:
                file_data['mods'].append(mod_data.url)
            file_data[mod_name] = json_data[mod_name]

            json_data = file_data

    with open(installer_path, "w") as installer_file:
        json.dump(json_data, installer_file)

    return json_data


@router.post("/upload-file/")
async def create_upload_file(file: UploadFile):
    tb_app: App = get_app()
    if tb_app.debug:
        do = False
        try:
            tb_app.load_mod(file.filename.split(".py")[0])
        except ModuleNotFoundError:
            do = True

        if do:
            try:
                with open("./mods/" + file.filename, 'wb') as f:
                    while contents := file.file.read(1024 * 1024):
                        f.write(contents)
            except Exception:
                return {"res": "There was an error uploading the file"}
            finally:
                file.file.close()

            return {"res": f"Successfully uploaded {file.filename}"}
    return {"res": "not avalable"}


@router.get("/{path:path}")
def download_file(path: str):
    file_name = path
    TB_DIR = get_app().start_dir
    print(TB_DIR)
    if platform.system() == "Darwin" or platform.system() == "Linux":
        directory = file_name.split("/")
    else:
        directory = file_name.split("\\")
    print(directory)
    if len(directory) == 1:
        directory = file_name.split("%5")
    print(directory)
    get_logger().info(f"Request file {file_name}")

    if ".." in file_name:
        return {"message": f"unsupported operation .. "}

    if platform.system() == "Darwin" or platform.system() == "Linux":
        file_path = TB_DIR + "/" + file_name
    else:
        file_path = TB_DIR + "\\" + file_name

    print(file_path)
    if len(directory) > 1:
        directory = directory[0]

        if directory not in ["mods", "runnable", "tests", "data", "requirements", "pconfig", "utils", "installer"]:
            get_logger().warning(f"{file_path} not public")
            return JSONResponse(content={"message": f"directory not public {directory}"}, status_code=100)

        if directory == "tests":
            if platform.system() == "Darwin" or platform.system() == "Linux":
                file_path = "/".join(TB_DIR.split("/")[:-1]) + "/" + file_name
            else:
                file_path = "\\".join(TB_DIR.split("\\")[:-1]) + "\\" + file_name

    if os.path.exists(file_path):
        get_logger().info(f"Downloading from {file_path}")
        if os.path.isfile(file_path):
            return FileResponse(file_path, media_type='application/octet-stream', filename=file_name)
        return JSONResponse(content={"message": f"is directory", "files": os.listdir(file_path)}, status_code=201)
    else:
        get_logger().error(f"{file_path} not found")
        return JSONResponse(content={"message": "File not found"}, status_code=110)

# def mount_mod_files(app: FastAPI):
#     routes = [
#         Mount("/mods", StaticFiles(directory="./mods"), name="mods"),
#         Mount("/runnable", StaticFiles(directory="./runabel"), name="runnable"),
#         Mount("/requirements", StaticFiles(directory="./requirements"), name="requirements"),
#         Mount("/test", StaticFiles(directory="./mod_data"), name="data"),
#         Mount("/data", StaticFiles(directory="./../tests"), name="test")
#
#     ]
#     s_app = Starlette(routes=routes)
#
#     app.route("/install", routes, name="installer")
#     app.mount("/installer2", s_app, name="installer2")
#
#     # s_app der app hinzufügen unter /install Route
#     @app.middleware("http")
#     async def mount_static_files(request: Request, call_next):
#         if request.url.path.startswith("/install"):
#             # Weiterleitung an die Starlette-Anwendung
#             state = request.state
#             return await s_app()
#         # Weiterleitung an die nächste Middleware oder die Route-Handler-Funktion
#         return await call_next(request)
#
