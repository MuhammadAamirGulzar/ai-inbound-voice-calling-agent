from .base import FulfillmentAdapter, DispatchResult

class ManualRedirectAdapter(FulfillmentAdapter):
    def dispatch(self, order, restaurant) -> DispatchResult:
        target_url = getattr(restaurant, "fulfillment_target_url", None)
        
        if not target_url or not target_url.lower().startswith("http"):
            return DispatchResult(
                success=False,
                redirect_url=None,
                message="Invalid target URL. Connection refused or invalid protocol."
            )
            
        return DispatchResult(
            success=True,
            redirect_url=target_url,
            message="Manual redirect URL fetched successfully."
        )
