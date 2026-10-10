"""
FastAPI Request and Response Pydantic Schemas (Assignment 4).
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class InferenceRequest(BaseModel):
    x: List[List[float]] = Field(..., description="Node feature matrix of shape (num_nodes, num_features)")
    edge_index: List[List[int]] = Field(..., description="Graph connectivity of shape (2, num_edges)")
    target_nodes: Optional[List[int]] = Field(default=None, description="Subset of node IDs to return predictions for")


class PredictionItem(BaseModel):
    node_id: int
    predicted_class: int
    class_name: str
    probabilities: List[float]


class InferenceResponse(BaseModel):
    model_version: str
    num_nodes: int
    num_edges: int
    predictions: List[PredictionItem]
    warnings: Optional[List[str]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str]
    in_channels: Optional[int]
    num_classes: Optional[int]
