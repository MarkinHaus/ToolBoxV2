from toolboxv2 import App, Code, RequestData
import json
import uuid
from toolboxv2.mods.videoFlow.engine.project_manager import ProjectManager
from toolboxv2.mods.videoFlow.engine.config import CostTracker

def register_api_endpoints(app: App):
    @app.export(api=True, mod_name="videoFlow", route="/register", method="POST")
    async def register(request_data: RequestData) -> dict:
        username = request_data.get("username")
        password = request_data.get("password")

        if not username or not password:
            return {"status": "error", "message": "Username and password are required.", "status_code": 400}

        users_db_path = ProjectManager(CostTracker()).base_dir / "users.json"
        users = {}
        if users_db_path.exists():
            with open(users_db_path, 'r') as f:
                users = json.load(f)

        if username in users:
            return {"status": "error", "message": "Username already exists.", "status_code": 400}

        hashed_password = Code.hash_password(password)
        users[username] = {"password": hashed_password, "user_id": str(uuid.uuid4())}

        with open(users_db_path, 'w') as f:
            json.dump(users, f, indent=2)

        return {"status": "success", "message": "User registered successfully.", "status_code": 200}

    @app.export(api=True, mod_name="videoFlow", route="/login", method="POST")
    async def login(request_data: RequestData) -> dict:
        username = request_data.get("username")
        password = request_data.get("password")

        if not username or not password:
            return {"status": "error", "message": "Username and password are required.", "status_code": 400}

        users_db_path = ProjectManager(CostTracker()).base_dir / "users.json"
        users = {}
        if users_db_path.exists():
            with open(users_db_path, 'r') as f:
                users = json.load(f)

        user_info = users.get(username)
        if not user_info or not Code.verify_password(password, user_info["password"]):
            return {"status": "error", "message": "Invalid username or password.", "status_code": 401}

        return {"status": "success", "message": "Login successful.", "user_id": user_info["user_id"], "status_code": 200}
