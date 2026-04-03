from toolboxv2 import App, RequestData

def register_api_endpoints(app: App):
    @app.export(api=True, mod_name="videoFlow", route="/credits/{user_id}", method="GET")
    async def get_credits(request_data: RequestData, user_id: str) -> dict:
        # Placeholder for a credit system
        # In a real application, this would query a database for user credits
        
        # For now, everyone has 100 credits
        credits = 100
        return {"user_id": user_id, "credits": credits, "status_code": 200}

    @app.export(api=True, mod_name="videoFlow", route="/deduct_credits", method="POST")
    async def deduct_credits(request_data: RequestData) -> dict:
        user_id = request_data.get("user_id")
        amount = request_data.get("amount")

        if not user_id:
            return {"status": "error", "message": "Unauthorized: User ID not found in session.", "status_code": 401}
        if not amount or not isinstance(amount, (int, float)) or amount <= 0:
            return {"status": "error", "message": "Invalid amount specified.", "status_code": 400}

        # Placeholder for deducting credits
        # In a real application, this would update a database
        
        # For now, always succeed
        return {"status": "success", "message": "Credits deducted successfully.", "status_code": 200}
