#!/usr/bin/env python3
import requests
import json

# Configuration
API_URL = "http://localhost:8000/quiz"
EMAIL = "21f3002378@ds.study.iitm.ac.in"
SECRET = "elephant"
QUIZ_URL = "https://tds-llm-analysis.s-anand.net/demo"

# Payload
payload = {
    "email": EMAIL,
    "secret": SECRET,
    "url": QUIZ_URL
}

print("Sending request to:", API_URL)
print("Payload:", json.dumps(payload, indent=2))

# Send request
response = requests.post(API_URL, json=payload)

print("\nResponse Status:", response.status_code)
print("Response Body:")
print(json.dumps(response.json(), indent=2))