import requests
import random
import string
import json
import time

BASE_URL = "https://well360-backend.onrender.com"

def print_header(title):
    print(f"\n{'='*50}\n{title}\n{'='*50}")

def print_result(name, passed, details=""):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} - {name}")
    if details:
        print(f"   Details: {details}")

def generate_random_email():
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"test_user_{random_str}@example.com"

def check_endpoint(url, description):
    try:
        response = requests.get(url, timeout=10)
        passed = response.status_code == 200
        print_result(description, passed, f"Status: {response.status_code}")
        return passed
    except Exception as e:
        print_result(description, False, f"Error: {str(e)}")
        return False

def main():
    print_header("STARTING API SYSTEM CHECK")
    print(f"Target: {BASE_URL}")

    # 1. PUBLIC HEALTH CHECKS
    print_header("1. PUBLIC HEALTH CHECKS")
    
    # Check Main API Status
    check_endpoint(f"{BASE_URL}/api-status", "Main API Status Checking")
    
    # Check Hydration Module
    check_endpoint(f"{BASE_URL}/hydration/health", "Hydration Module Health")

    # Check Mental Health Module
    check_endpoint(f"{BASE_URL}/mental-health/status", "Mental Health Module Status")

    # 2. AUTHENTICATION FLOW
    print_header("2. AUTHENTICATION & PERSISTENCE")
    
    email = generate_random_email()
    password = "TestPassword123!"
    token = None
    
    # Register
    print(f"Attempting to register user: {email}")
    try:
        reg_payload = {"email": email, "password": password, "age": 25, "weight": 70, "height": 175, "gender": "Male"}
        reg_resp = requests.post(f"{BASE_URL}/auth/register", json=reg_payload, timeout=10)
        
        if reg_resp.status_code == 200:
            print_result("User Registration", True)
        else:
            print_result("User Registration", False, f"Status: {reg_resp.status_code}, Body: {reg_resp.text}")
    except Exception as e:
        print_result("User Registration", False, f"Exception: {str(e)}")

    # Login
    print("Attempting login...")
    try:
        login_data = {"username": email, "password": password} # OAuth2 form data
        login_resp = requests.post(f"{BASE_URL}/auth/login", data=login_data, timeout=10)
        
        if login_resp.status_code == 200:
            token = login_resp.json().get("access_token")
            print_result("User Login", True, "Token acquired")
        else:
            print_result("User Login", False, f"Status: {login_resp.status_code}, Body: {login_resp.text}")
    except Exception as e:
        print_result("User Login", False, f"Exception: {str(e)}")

    # 3. PROTECTED ROUTES
    if token:
        print_header("3. PROTECTED MODULE ACCESS")
        headers = {"Authorization": f"Bearer {token}"}
        
        # Check Profile (Auth verification)
        try:
            profile_resp = requests.get(f"{BASE_URL}/auth/profile", headers=headers, timeout=10)
            if profile_resp.status_code == 200:
                print_result("Get Profile Data", True, f"Email: {profile_resp.json().get('email')}")
            else:
                print_result("Get Profile Data", False, f"Status: {profile_resp.status_code}")
        except Exception as e:
            print_result("Get Profile Data", False, str(e))

        # Check Hydration Weather Endpoint (Requires Token)
        try:
            weather_resp = requests.get(f"{BASE_URL}/weather/current?lat=40.71&lon=-74.00", headers=headers, timeout=10)
            if weather_resp.status_code == 200:
                print_result("Hydration (Weather) Access", True)
            else:
                print_result("Hydration (Weather) Access", False, f"Status: {weather_resp.status_code}")
        except Exception as e:
            print_result("Hydration (Weather) Access", False, str(e))

    else:
        print("\n⚠️ Skipping protected route tests because login failed.")

if __name__ == "__main__":
    main()
