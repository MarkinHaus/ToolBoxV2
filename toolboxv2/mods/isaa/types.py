
from pydantic import BaseModel, Field


class Task(BaseModel):
    use: str = Field(..., description="The type of task to be executed (agent, chain, or tool)")
    name: str = Field(..., description="The name of the task")
    args: str = Field(..., description="The arguments for the task, must include the var $user-input")
    return_key: str = Field(..., description="The key under which the task's result will be stored")


class TaskChain(BaseModel):
    name: str = Field(..., description="The name of the task chain")
    tasks: list[Task] = Field(..., description="An array of tasks to be executed in order")
    description: str | None = Field(None, description="Optional description or additional information about the task chain")
