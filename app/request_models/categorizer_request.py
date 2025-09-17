from pydantic import BaseModel, Field


class CategorizerRequest(BaseModel):
    product_list: list = Field(..., description="List of product descriptions for categorization")
    model_name: str = Field(default='RF_Categorizer_Model', description="Name of the model to use for inference")
