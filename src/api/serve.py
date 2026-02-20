from fastapi import FastAPI
from pydantic import BaseModel
from src.models.predict import predict

app = FastAPI()

class OrderInput(BaseModel):
    delivery_delay_days: float
    price: float
    freight_value: float
    seller_historical_risk: float

@app.post("/predict")
def predict_order(order: OrderInput):
    result = predict(order.dict())
    return result