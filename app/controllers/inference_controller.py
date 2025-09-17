from fastapi import APIRouter, HTTPException

from app.services.inference_service import InferenceService
from app.responses.categorizer_response import CategorizerResponse
from app.request_models.categorizer_request import CategorizerRequest


api = APIRouter()
__inference_service = InferenceService()
__inference_service.setup()


@api.post(path="/v1/categorizer/inference")
def get_category_inference(request: CategorizerRequest) -> CategorizerResponse:
    product_list = request.product_list
    category_prediction = __inference_service.predict(product_list=product_list)
    if category_prediction is None:
        raise HTTPException(status_code=404, detail='Error during category prediction')

    output = CategorizerResponse(
        errors=False,
        categories=category_prediction.tolist(),
        model_name=__inference_service.model_name
    )

    return output
