"""Abstract base class defining the provider interface for e-Challan lookups."""

from __future__ import annotations

from abc import ABC, abstractmethod
from ml.challan.schemas import ChallanCheckResponse


class ChallanProvider(ABC):
    """Abstract interface that all e-Challan providers must implement."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Identifier of this provider implementation."""
        pass

    @abstractmethod
    async def check_challans(self, vehicle_number: str) -> ChallanCheckResponse:
        """
        Fetch citation records for the given vehicle number.
        Returns standardized ChallanCheckResponse.
        """
        pass
