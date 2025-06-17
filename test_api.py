import requests
import json

def test_api():
    # Base URL
    base_url = "http://localhost:8000"
    
    # Test root endpoint
    print("\nTesting root endpoint...")
    response = requests.get(f"{base_url}/")
    print(f"Response: {response.json()}")
    
    # Test refresh endpoint
    print("\nRefreshing news database...")
    response = requests.post(f"{base_url}/refresh")
    print(f"Response: {response.json()}")
    
    # Test analyze endpoint
    print("\nAnalyzing news about AI...")
    query = {
        "query": "What are the latest developments in AI?",
        "max_results": 3
    }
    response = requests.post(f"{base_url}/analyze", json=query)
    print("\nAnalysis Results:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_api() 