import json
import time
import requests
from typing import Dict, Any, Optional

class MockAPI:
    """
    A mock API service that simulates responses for specific types of information requests.
    This can be used to simulate external service calls without actual API dependencies.
    """
    
    def __init__(self):
        # Initialize with some mock data for tax and user information
        self.mock_data = {
            "tax_rates": {
                "vat_standard": 22,
                "vat_reduced": 10,
                "vat_super_reduced": 4,
                "income_tax_brackets": [
                    {"bracket": "0-15,000", "rate": 23},
                    {"bracket": "15,001-28,000", "rate": 27},
                    {"bracket": "28,001-50,000", "rate": 38},
                    {"bracket": "50,001+", "rate": 43}
                ]
            },
            "deadlines": {
                "vat_payment": "16th of each month",
                "annual_tax_return": "November 30th",
                "quarterly_vat": ["May 16th", "August 16th", "November 16th", "February 16th"]
            },
            "user_profiles": {
                "123456": {
                    "name": "Mario Rossi",
                    "vat_number": "IT12345678901",
                    "regime": "forfettario",
                    "annual_income": "45000",
                    "activity_code": "62.01.00"
                }
            }
        }
        
        # External API endpoint for invoice data
        # This is a real API endpoint that returns sample invoice data
        self.invoice_api_url = "https://679fecaf24322f8329c4ea4e.mockapi.io/invoices"
    
    def call(self, endpoint: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Simulate an API call to a specific endpoint with parameters.
        This is the main entry point for all API calls in the system.
        
        Args:
            endpoint: The API endpoint to call (tax_rates, deadlines, user_profile, or invoices)
            parameters: Optional parameters for the API call
            
        Returns:
            A dictionary containing the mock API response with data and status code
        """
        # Simulate network delay for a more realistic API experience
        time.sleep(0.5)
        
        if parameters is None:
            parameters = {}
            
        # Route to the appropriate mock endpoint handler
        if endpoint == "tax_rates":
            return self._get_tax_rates(parameters)
        elif endpoint == "deadlines":
            return self._get_deadlines(parameters)
        elif endpoint == "user_profile":
            return self._get_user_profile(parameters)
        elif endpoint == "invoices":
            # For invoice data, we call the external MockAPI.io service
            return self._get_invoices(parameters)
        else:
            return {"error": "Endpoint not found", "status": 404}
    
    def _get_tax_rates(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get tax rate information from mock data.
        
        Args:
            parameters: Dictionary containing parameters like "type" to filter tax rates
            
        Returns:
            Dictionary with tax rate data and status code
        """
        rate_type = parameters.get("type")
        if rate_type:
            if rate_type in self.mock_data["tax_rates"]:
                return {"data": self.mock_data["tax_rates"][rate_type], "status": 200}
            else:
                return {"error": f"Tax rate type '{rate_type}' not found", "status": 404}
        return {"data": self.mock_data["tax_rates"], "status": 200}
    
    def _get_deadlines(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get tax deadline information from mock data.
        
        Args:
            parameters: Dictionary containing parameters like "type" to filter deadlines
            
        Returns:
            Dictionary with deadline data and status code
        """
        deadline_type = parameters.get("type")
        if deadline_type:
            if deadline_type in self.mock_data["deadlines"]:
                return {"data": self.mock_data["deadlines"][deadline_type], "status": 200}
            else:
                return {"error": f"Deadline type '{deadline_type}' not found", "status": 404}
        return {"data": self.mock_data["deadlines"], "status": 200}
    
    def _get_user_profile(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get user profile information from mock data.
        
        Args:
            parameters: Dictionary containing "user_id" parameter
            
        Returns:
            Dictionary with user profile data and status code
        """
        user_id = parameters.get("user_id")
        if not user_id:
            return {"error": "User ID is required", "status": 400}
            
        if user_id in self.mock_data["user_profiles"]:
            return {"data": self.mock_data["user_profiles"][user_id], "status": 200}
        else:
            return {"error": f"User profile with ID '{user_id}' not found", "status": 404}
            
    def _get_invoices(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get invoice information from the external MockAPI.io service.
        This demonstrates integration with a real external API.
        
        Args:
            parameters: Dictionary that may contain "invoice_id" or "customer_id"
            
        Returns:
            Dictionary with invoice data and status code
        """
        try:
            invoice_id = parameters.get("invoice_id")
            customer_id = parameters.get("customer_id")
            
            # Build the URL based on the parameters
            url = self.invoice_api_url
            
            # If invoice_id is provided, get a specific invoice by appending ID to URL
            if invoice_id:
                url = f"{url}/{invoice_id}"
                
            # Make the actual HTTP request to the external API
            response = requests.get(url)
            
            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                
                # If customer_id is provided and we have a list of invoices, filter by customer
                if customer_id and isinstance(data, list):
                    filtered_data = [invoice for invoice in data if invoice.get("customerId") == customer_id]
                    return {"data": filtered_data, "status": 200}
                    
                # Return the data as is if no filtering is needed
                return {"data": data, "status": 200}
            else:
                # Return the error details if the request failed
                return {"error": f"API request failed with status code: {response.status_code}", "status": response.status_code}
                
        except Exception as e:
            # Handle any exceptions during the API call
            return {"error": f"Error accessing invoice API: {str(e)}", "status": 500} 