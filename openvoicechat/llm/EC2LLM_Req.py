import requests
import json
from typing import Dict, Any, Optional, Union

class EC2LLMClient:
    """
    Client for interacting with an LLM service running on an EC2 instance.
    """
    
    def __init__(self, base_url: str, timeout: int = 60):
        """
        Initialize the EC2 LLM client.
        
        Args:
            base_url: The base URL of the EC2 instance 
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def query_llm(self, 
                 input_text: str, 
                 endpoint: str = "/chat", 
                 minibot_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Send a query to the LLM endpoint and get the response.
        
        Args:
            input_text: The user input text to send to the LLM
            endpoint: The API endpoint to use (default: "/chat")
            minibot_args: Optional arguments for the minibot configuration
            
        Returns:
            The JSON response from the LLM service
        """
        url = f"{self.base_url}{endpoint}"
        
        payload = {
            "input_text": input_text
        }
        
        if minibot_args:
            payload["minibot_args"] = minibot_args
            
        try:
            response = self.session.get(
                url,
                params=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "failed"}
    
    def stream_query(self, 
                    input_text: str, 
                    endpoint: str = "/chat/stream", 
                    minibot_args: Optional[Dict[str, Any]] = None) -> Union[Dict[str, Any], str]:
        """
        Send a streaming query to the LLM endpoint and yield responses as they come.
        
        Args:
            input_text: The user input text to send to the LLM
            endpoint: The API endpoint to use (default: "/chat/stream")
            minibot_args: Optional arguments for the minibot configuration
            
        Yields:
            Response chunks from the LLM service
        """
        url = f"{self.base_url}{endpoint}"
        
        payload = {
            "input_text": input_text
        }
        
        if minibot_args:
            payload["minibot_args"] = minibot_args
            
        try:
            response = self.session.get(
                url,
                params=payload,
                stream=True,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    try:
                        yield json.loads(chunk.decode('utf-8'))
                    except json.JSONDecodeError:
                        yield chunk.decode('utf-8')
                        
        except requests.exceptions.RequestException as e:
            yield {"error": str(e), "status": "failed"}

# Example usage
if __name__ == "__main__":
    # Initialize the client with your EC2 instance URL
    ec2_url = "http://ec2-12-345-67-890.compute-1.amazonaws.com:8000"  
    client = EC2LLMClient(ec2_url)
    
    # Example minibot args (same as in your code)
    minibot_args = {
    "agent_name": "Alice",
    "organization_name": "Medicare",
    "total_products": 1,
    "product_names": ["Hospital Insurance"],
    "product_descriptions": [
        "Covers inpatient care in hospitals, skilled nursing facility care, hospice care, and some home health services."
    ],
    "product_features": [
        "Free for Citizens above 70"
    ],
}
    
    # Standard request
    response = client.query_llm("Tell me about Medcare", minibot_args=minibot_args)
    print(f"LLM Response: {response}")
    
    # Streaming request example
    print("\nStreaming response:")
    for chunk in client.stream_query("What features does the Medicare have?", minibot_args=minibot_args):
        if isinstance(chunk, dict) and "error" in chunk:
            print(f"Error: {chunk['error']}")
            break
        print(chunk, end="", flush=True)