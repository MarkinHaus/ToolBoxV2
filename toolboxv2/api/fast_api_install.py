import os
import platform

from fastapi import APIRouter, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from toolboxv2 import get_logger, App, get_app

router = APIRouter(
    prefix="/install",
)

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

@router.get("/{file_name}")
def download_file(file_name: str):
    TB_DIR = get_app().start_dir
    if platform.system() == "Darwin" or platform.system() == "Linux":
        directory = file_name.split("/")
    else:
        directory = file_name.split("\\")
    if len(directory) == 1:
        directory = file_name.split("%5")
    get_logger().info(f"Request file {file_name}")

    if ".." in file_name:
        return {"message": f"unsupported operation .. "}

    if platform.system() == "Darwin" or platform.system() == "Linux":
        file_path = TB_DIR + "/" + file_name
    else:
        file_path = TB_DIR + "\\" + file_name
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
        return FileResponse(file_path, media_type='application/octet-stream', filename=file_name)
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
