from langchain.tools import BaseTool
from typing import Optional, Dict, Any, List
from pydantic import Field, BaseModel

class ValidateZipcodeTool(BaseTool):
    name: str = "validate_zipcode"
    description: str = "Validates if a zipcode is valid and returns information about the area"
    
    def _run(self, zipcode: str) -> str:
        # Mock implementation
        return f"Zipcode {zipcode} is valid. It's in Los Angeles County, California."
    
    async def _arun(self, zipcode: str) -> str:
        return self._run(zipcode)

class GetProductInfoTool(BaseTool):
    name: str = "get_product_info"
    description: str = "Gets detailed information about a specific product"
    products: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    def _run(self, product_name: str) -> str:
        # Search for the product in the provided product dictionary
        for name, details in self.products.items():
            if product_name.lower() in name.lower():
                return f"{name}: {details.get('description', '')} {details.get('feature', '')}"
        return "Product information not found."
    
    async def _arun(self, product_name: str) -> str:
        return self._run(product_name)

# Define a base class for user info tools
class UserInfoTool(BaseTool):
    user_info_instance: Any = Field(exclude=True)

class UpdateNameTool(UserInfoTool):
    name: str = "update_name"
    description: str = "Updates the user's name"

    def _run(self, name: str) -> str:
        self.user_info_instance.name = name
        return f"Updated user name to {name}"
    
    async def _arun(self, name: str) -> str:
        return self._run(name)

class UpdateAgeTool(UserInfoTool):
    name: str = "update_age"
    description: str = "Updates the user's age"

    def _run(self, age: str) -> str:
        try:
            self.user_info_instance.age = int(age)
            return f"Updated user age to {age}"
        except ValueError:
            return f"Could not update age: {age} is not a valid number"
    
    async def _arun(self, age: str) -> str:
        return self._run(age)

class UpdateCityTool(UserInfoTool):
    name: str = "update_city"
    description: str = "Updates the user's city"

    def _run(self, city: str) -> str:
        self.user_info_instance.city = city
        return f"Updated user city to {city}"
    
    async def _arun(self, city: str) -> str:
        return self._run(city)

class UpdateZipcodeTool(UserInfoTool):
    name: str = "update_zipcode"
    description: str = "Updates the user's zipcode"

    def _run(self, zipcode: str) -> str:
        self.user_info_instance.zipcode = zipcode
        return f"Updated user zipcode to {zipcode}"
    
    async def _arun(self, zipcode: str) -> str:
        return self._run(zipcode)

class UpdateInterestTool(UserInfoTool):
    name: str = "update_interest"
    description: str = "Updates the user's interest level"

    def _run(self, interest: str) -> str:
        if "yes" in interest.lower() or "interested" in interest.lower():
            self.user_info_instance.interest_level = "Interested"
        elif "no" in interest.lower() or "not" in interest.lower():
            self.user_info_instance.interest_level = "Not interested"
        else:
            self.user_info_instance.interest_level = "Maybe interested"
        return f"Updated user interest to {self.user_info_instance.interest_level}"
    
    async def _arun(self, interest: str) -> str:
        return self._run(interest)

class VerifyInfoTool(UserInfoTool):
    name: str = "verify_info"
    description: str = "Marks a field as verified"

    def _run(self, field: str) -> str:
        if field == "name":
            self.user_info_instance.verified['name'] = True
        elif field == "age":
            self.user_info_instance.verified['age'] = True
        elif field == "city":
            self.user_info_instance.verified['city'] = True
        elif field == "zipcode":
            self.user_info_instance.verified['zipcode'] = True
        elif field == "interest":
            self.user_info_instance.verified['interest'] = True
        return f"Verified user {field}"
    
    async def _arun(self, field: str) -> str:
        return self._run(field)

def create_user_info_tools(user_info_instance):
    """Create tools that interact with the UserInformation instance."""
    return [
        UpdateNameTool(user_info_instance=user_info_instance),
        UpdateAgeTool(user_info_instance=user_info_instance),
        UpdateCityTool(user_info_instance=user_info_instance),
        UpdateZipcodeTool(user_info_instance=user_info_instance),
        UpdateInterestTool(user_info_instance=user_info_instance),
        VerifyInfoTool(user_info_instance=user_info_instance),
    ]