from abc import ABC, abstractmethod
from typing import Optional
from dataclasses import dataclass

@dataclass
class DispatchResult:
    success: bool
    redirect_url: Optional[str]
    message: str

class FulfillmentAdapter(ABC):
    @abstractmethod
    def dispatch(self, order, restaurant) -> DispatchResult:
        pass
