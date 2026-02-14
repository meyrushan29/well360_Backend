"""
Test Script for Lip Analysis Improvements
Tests all new features to ensure everything works correctly
"""
import requests
import base64
import json
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:5000"  # Change to your deployment URL
TOKEN = None  # Will be filled after login

def login(email, password):
    """Login and get token"""
    global TOKEN
    response = requests.post(
        f"{BASE_URL}/auth/login",
        data={"username": email, "password": password}
    )
    if response.status_code == 200:
        TOKEN = response.json()["access_token"]
        print("✅ Login successful!")
        return True
    else:
        print(f"❌ Login failed: {response.text}")
        return False

def test_lip_prediction(image_path):
    """Test enhanced lip prediction with advanced features"""
    print("\n🧪 Testing Lip Prediction with Advanced Features...")
    
    # Read and encode image
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    
    response = requests.post(
        f"{BASE_URL}/predict/lip",
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Content-Type": "application/json"
        },
        json={"image_base64": img_data}
    )
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Prediction successful!")
        print(f"\n📊 Results:")
        print(f"  Prediction: {result.get('prediction')}")
        print(f"  Hydration Score: {result.get('hydration_score')}/100")
        print(f"  Confidence: {result.get('confidence', 0)*100:.1f}%")
        
        # Check for advanced analysis
        advanced = result.get('advanced_analysis', {})
        if advanced:
            print(f"\n🔬 Advanced Analysis:")
            print(f"  Quality Score: {advanced.get('quality_score')}/100")
            print(f"  Lip Detected: {advanced.get('lip_detected')}")
            print(f"  Crack Severity: {advanced.get('crack_severity')}/100")
            print(f"  Texture Roughness: {advanced.get('texture_roughness'):.1f}")
        else:
            print("\n⚠️ Advanced analysis not available (module may not be loaded)")
        
        # Check XAI
        xai = result.get('xai_description')
        if xai:
            print(f"\n💡 AI Reasoning:")
            print(f"  {xai[:200]}...")
        
        return True
    else:
        print(f"❌ Prediction failed: {response.text}")
        return False

def test_lip_trends():
    """Test lip trends endpoint"""
    print("\n📈 Testing Lip Trends Endpoint...")
    
    response = requests.get(
        f"{BASE_URL}/history/lip-trends",
        headers={"Authorization": f"Bearer {TOKEN}"}
    )
    
    if response.status_code == 200:
        data = response.json()
        summary = data.get('summary', {})
        trend_data = data.get('trend_data', [])
        
        print("✅ Trends endpoint working!")
        print(f"\n📊 Summary:")
        print(f"  Total Scans: {summary.get('total_scans', 0)}")
        print(f"  Average Score: {summary.get('avg_score', 0):.1f}/100")
        print(f"  Improvement: {summary.get('improvement', 0):+.1f} points")
        print(f"  Dehydrated Count: {summary.get('dehydrated_count', 0)}")
        print(f"  Normal Count: {summary.get('normal_count', 0)}")
        print(f"  Best Score: {summary.get('best_score', 0)}")
        print(f"  Worst Score: {summary.get('worst_score', 0)}")
        
        print(f"\n📅 Data Points: {len(trend_data)}")
        if trend_data:
            print(f"  First: {trend_data[0].get('date')} - Score: {trend_data[0].get('score')}")
            print(f"  Latest: {trend_data[-1].get('date')} - Score: {trend_data[-1].get('score')}")
        
        return True
    else:
        print(f"❌ Trends failed: {response.text}")
        return False

def test_feature_extraction():
    """Test if advanced features module is working"""
    print("\n🔬 Testing Feature Extraction Module...")
    
    try:
        from hydration.lip_feature_extractor import (
            extract_all_features,
            calculate_image_quality_score
        )
        print("✅ Feature extraction module imported successfully!")
        
        # Test with a sample image
        from PIL import Image
        import numpy as np
        
        # Create a test image
        test_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        
        # Test feature extraction
        features = extract_all_features(test_img, auto_enhance=True)
        quality = calculate_image_quality_score(features['features'])
        
        print(f"  Quality Score: {quality:.1f}/100")
        print(f"  Lip Detected: {features['features'].get('lip_detected')}")
        print(f"  Crack Severity: {features['features'].get('crack_severity_score', 0):.1f}")
        
        return True
    except Exception as e:
        print(f"❌ Feature extraction test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("🧪 LIP ANALYSIS IMPROVEMENTS - TEST SUITE")
    print("=" * 60)
    
    # Get credentials
    email = input("\n📧 Enter your email: ").strip()
    password = input("🔐 Enter your password: ").strip()
    
    if not login(email, password):
        return
    
    # Test feature extraction (backend only)
    print("\n" + "=" * 60)
    test_feature_extraction()
    
    # Test lip prediction
    print("\n" + "=" * 60)
    image_path = input("\n📷 Enter path to test lip image (or press Enter to skip): ").strip()
    if image_path and Path(image_path).exists():
        test_lip_prediction(image_path)
    else:
        print("⏭️ Skipping lip prediction test")
    
    # Test trends
    print("\n" + "=" * 60)
    test_lip_trends()
    
    print("\n" + "=" * 60)
    print("✅ TESTING COMPLETE!")
    print("=" * 60)
    print("\n📝 Results Summary:")
    print("  If all tests passed, deployment is ready!")
    print("  If any failed, check backend logs for errors")

if __name__ == "__main__":
    main()
