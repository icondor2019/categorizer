from pydantic import BaseModel


class CategorizerResponse(BaseModel):
    errors: bool
    categories: list
    model_name: str
