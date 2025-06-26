from pydantic import BaseModel


class InputData(BaseModel):
    point: str
    params: dict
