import logging
from .base import FulfillmentAdapter
from .manual_redirect import ManualRedirectAdapter

logger = logging.getLogger(__name__)

# Register all adapters here
_ADAPTERS = {
    "manual_redirect": ManualRedirectAdapter
}

def get_adapter(integration_type: str) -> FulfillmentAdapter:
    adapter_cls = _ADAPTERS.get(integration_type)
    if not adapter_cls:
        logger.warning(
            f"Fulfillment integration type '{integration_type}' is not recognized. "
            f"Falling back to 'manual_redirect'."
        )
        return ManualRedirectAdapter()
    return adapter_cls()
