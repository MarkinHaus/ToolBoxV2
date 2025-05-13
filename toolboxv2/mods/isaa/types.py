
from pydantic import BaseModel, Field


class SupTask(BaseModel):
    """SupTask"""
    task: str = Field(..., description="The Sub Task it self withe all information and instructions."
                                       " From the main task specific for this sub-task.")
    tools: bool | None = Field(..., description="If the subtask needs tool usage")

class IsToolTask(BaseModel):
    """test if is a tool task """
    tools: bool = Field(..., description="If the task needs tool usage")


class TaskPlan(BaseModel):
    """TaskPlan"""
    sub_tasks: list[SupTask] = Field(..., description="A list of sub tasks to accomplish the main tasks")


class PlanEval(BaseModel):
    """evaluate if is possible to carry out the task effectively"""
    possible: bool = Field(..., description="Dos the sub tasks led to the final task completion?")


class TaskDone(BaseModel):
    """evaluate if the task accomplished?"""
    done: bool = Field(..., description="is the task accomplished?")


class TaskComplexity(BaseModel):
    """evaluate the complexity of the task between 0 and 10"""
    complexity: int = Field(..., description="complexity of the task")
    context: int = Field(...,
                         description="complexity of the needed context to solve the task 0 to 3 (HISTORY=1) (WEB=1) (ALL=3)")


class Task(BaseModel):
    use: str = Field(..., description="The type of task to be executed (agent, chain, or tool)")
    name: str = Field(..., description="The name of the task")
    args: str = Field(..., description="The arguments for the task, must include the var $user-input")
    return_key: str = Field(..., description="The key under which the task's result will be stored")


class TaskChain(BaseModel):
    name: str = Field(..., description="The name of the task chain")
    tasks: list[Task] = Field(..., description="An array of tasks to be executed in order")
    dis: str | None = Field(None, description="Optional description or additional information about the task chain")
