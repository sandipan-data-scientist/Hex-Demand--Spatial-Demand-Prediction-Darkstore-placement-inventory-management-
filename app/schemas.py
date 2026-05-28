# Defines the shape of API request and response bodies using Pydantic.
# Pydantic validates incoming JSON automatically and raises clear errors
# if the types do not match, so you never have to write manual validation.

from pydantic import BaseModel, Field
from typing import List


class PredictRequest(BaseModel):
    lag_1: float = Field(
        ...,
        ge=0,
        description="Number of orders in the previous hour for this hexagon"
    )
    hour: int = Field(
        ...,
        ge=0,
        le=23,
        description="Hour of day (0 to 23)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "lag_1": 12.0,
                "hour": 20
            }
        }


class PredictResponse(BaseModel):
    predicted_demand: float = Field(
        ...,
        description="Predicted number of orders for the next period"
    )
    demand_tier: str = Field(
        ...,
        description="Demand classification: High, Medium, or Low"
    )


class BatchPredictRequest(BaseModel):
    records: List[PredictRequest] = Field(
        ...,
        description="List of hexagon-hour records to predict"
    )


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]