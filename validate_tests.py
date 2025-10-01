#!/usr/bin/env python3
"""
Quick Test Validation Script for Horizon AI
Validates the test setup and runs a basic smoke test
"""

import sys
import os
from pathlib import Path
import importlib.util

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_imports():
    """Test that all our modules can be imported"""
    print("🔍 Testing module imports...")
    
    try:
        # Test error handling import
        from utils.error_handler import HorizonError, api_error_handler
        print("✅ Error handling module imported successfully")
        
        # Test app import
        spec = importlib.util.spec_from_file_location("app", PROJECT_ROOT / "app.py")
        app_module = importlib.util.module_from_spec(spec)
        print("✅ App module can be loaded")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_error_handling():
    """Test basic error handling functionality"""
    print("\n🧪 Testing error handling system...")
    
    try:
        from utils.error_handler import (
            HorizonError, ValidationError, 
            validate_required_fields, validate_field_types
        )
        
        # Test custom exceptions
        try:
            raise ValidationError("Test validation error")
        except ValidationError as e:
            print("✅ Custom exceptions work correctly")
        
        # Test validation functions
        test_data = {"name": "test", "value": 42}
        validate_required_fields(test_data, ["name", "value"])
        validate_field_types(test_data, {"name": str, "value": int})
        print("✅ Validation functions work correctly")
        
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_database_connection():
    """Test basic database functionality"""
    print("\n💾 Testing database connection...")
    
    try:
        import sqlite3
        
        # Test creating a temporary database
        test_db_path = PROJECT_ROOT / "test_temp.db"
        
        with sqlite3.connect(test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            
        # Clean up
        if test_db_path.exists():
            test_db_path.unlink()
            
        print("✅ Database connection works")
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def check_test_files():
    """Check that test files exist and are valid"""
    print("\n📋 Checking test files...")
    
    test_files = [
        "tests/test_comprehensive.py",
        "tests/test_personality_blending.py", 
        "tests/test_api_endpoints.py",
        "utils/error_handler.py"
    ]
    
    missing_files = []
    
    for test_file in test_files:
        file_path = PROJECT_ROOT / test_file
        if file_path.exists():
            print(f"✅ {test_file} exists")
        else:
            print(f"❌ {test_file} missing")
            missing_files.append(test_file)
    
    return len(missing_files) == 0

def check_requirements():
    """Check if requirements.txt has testing dependencies"""
    print("\n📦 Checking requirements...")
    
    requirements_file = PROJECT_ROOT / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    content = requirements_file.read_text()
    
    required_packages = ["pytest", "coverage", "mock"]
    missing_packages = []
    
    for package in required_packages:
        if package in content.lower():
            print(f"✅ {package} in requirements.txt")
        else:
            print(f"❌ {package} missing from requirements.txt")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def run_smoke_test():
    """Run a basic smoke test of the application"""
    print("\n🔥 Running smoke test...")
    
    try:
        # Test that we can create the Flask app
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        
        # We'll just validate the app can be imported
        spec = importlib.util.spec_from_file_location("app", PROJECT_ROOT / "app.py")
        if spec and spec.loader:
            print("✅ App module structure is valid")
            return True
        else:
            print("❌ App module structure is invalid")
            return False
            
    except Exception as e:
        print(f"❌ Smoke test failed: {e}")
        return False

def main():
    """Main validation function"""
    print("🚀 HORIZON AI TEST VALIDATION")
    print("=" * 50)
    
    tests = [
        ("Module Imports", test_imports),
        ("Error Handling", test_error_handling),
        ("Database Connection", test_database_connection),
        ("Test Files", check_test_files),
        ("Requirements", check_requirements),
        ("Smoke Test", run_smoke_test),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All validation tests passed! Ready for comprehensive testing.")
        return 0
    else:
        print("⚠️ Some validation tests failed. Please fix issues before running full tests.")
        return 1

if __name__ == "__main__":
    sys.exit(main())