"""Schemas for standardized e-Challan request, response, and provider data models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChallanItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    challanNumber: str = Field(..., description="Unique identification number of the citation")
    date: str = Field(..., description="Date and time the violation was recorded")
    offence: str = Field(..., description="Description of the traffic or legal violation")
    location: str = Field(..., description="Location or highway where offense occurred")
    amount: float = Field(..., description="Total fine amount in INR")
    status: str = Field(..., description="Status of the challan, e.g. Pending, Paid")
    dueAmount: float = Field(..., description="Outstanding amount due in INR")


class ChallanCheckRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    vehicleNumber: str = Field(..., alias="vehicle_number", description="Indian vehicle registration number")


class ChallanCheckResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    success: bool = True
    vehicleNumber: str
    totalPending: int
    totalDue: float
    challans: list[ChallanItem] = Field(default_factory=list)
    provider: str
    disclaimer: str | None = None


class ChallanErrorDetail(BaseModel):
    code: str
    message: str


class ChallanErrorResponse(BaseModel):
    success: bool = False
    error: ChallanErrorDetail
