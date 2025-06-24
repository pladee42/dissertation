from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum

class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ChecklistItem(BaseModel):
    question: str = Field(..., description="The evaluation question")
    correct_answer: str = Field(..., description="Expected correct answer (Yes/No)")
    priority: Priority = Field(..., description="Priority level of this check")
    explanation: Optional[str] = Field(None, description="Why this check is important")

class Checklist(BaseModel):
    items: List[ChecklistItem] = Field(..., min_items=1, max_items=20)
    topic: str = Field(..., description="Email topic being evaluated")
    
    @field_validator('items')
    def validate_items(cls, v):
        if len(v) == 0:
            raise ValueError("Checklist must have at least one item")
        return v

class JudgmentResult(BaseModel):
    question: str
    yes_probability: float = Field(..., ge=0.0, le=1.0)
    no_probability: float = Field(..., ge=0.0, le=1.0)
    judgment: str = Field(..., pattern="^(Yes|No)$")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

class EvaluationResult(BaseModel):
    email_content: str
    checklist_results: List[JudgmentResult]
    overall_score: float = Field(..., ge=0.0, le=1.0)
    weighted_score: Optional[float] = Field(None, ge=0.0, le=1.0)
