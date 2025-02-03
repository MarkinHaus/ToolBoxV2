
if __name__ == "__main__": # claude-3-5-haiku-20241022
    from litellm import completion
    from pydantic import BaseModel
    class CalendarEvent(BaseModel):
        """Extract the event information."""
        name: str
        date: str
        participants: list[str]
    messages = [
        {"role": "user", "content": "Alice and Bob are going to a science fair on Friday."},
    ]

    resp = completion(
        model="groq/llama-3.1-8b-instant",
        messages=messages,
        response_format=CalendarEvent,
    )

    print(resp, type(resp))
    print("Received={}".format(resp))


    resp = completion(
        model="claude-3-5-haiku-20241022",
        messages=messages,
        response_format=CalendarEvent,
    )

    print(resp, type(resp))
    print("Received={}".format(resp))
