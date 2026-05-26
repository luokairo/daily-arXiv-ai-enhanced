from typing import List

from pydantic import BaseModel, Field

class Structure(BaseModel):
    is_relevant: bool = Field(description="whether the paper belongs to any target research direction")
    primary_direction_id: str = Field(description="id of the best matching target direction, or empty if irrelevant")
    matched_direction_ids: List[str] = Field(default_factory=list, description="ids of all matching target directions")
    subtopic_id: str = Field(description="stable id of the selected or newly created subtopic")
    subtopic_name: str = Field(description="short subtopic name")
    subtopic_description: str = Field(description="concise definition of the subtopic")
    is_new_subtopic: bool = Field(description="true if this subtopic is newly created instead of reused")
    classification_reason: str = Field(description="brief reason for direction and subtopic classification")
    tldr: str = Field(description="generate a too long; didn't read summary")
    motivation: str = Field(description="describe the motivation in this paper")
    method: str = Field(description="method of this paper")
    result: str = Field(description="result of this paper")
    conclusion: str = Field(description="conclusion of this paper")
