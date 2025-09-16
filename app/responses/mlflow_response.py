from pydantic import BaseModel


class RFCategorizerMetrics(BaseModel):
    accuracy: float
    f1_score_macro: float
    recall_macro: float
    precision_macro: float
    recall_abc: float
    recall_electricos: float
    recall_ferreteria: float
    recall_herramientas: float
    recall_pinturas: float
    recall_plomeria: float
