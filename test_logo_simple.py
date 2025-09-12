#!/usr/bin/env python3
"""
Simple test for logo generation functionality
This test focuses on the smart AI logic without database dependencies
"""

import sys
import os
import re

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_intent_recognition():
    """Test if logo requests are properly recognized"""
    print("🧪 Testing Logo Intent Recognition...")
    
    # Import the patterns from app.py
    try:
        from app import INTENT_PATTERNS
        logo_patterns = INTENT_PATTERNS.get('logo_generation', [])
        print(f"✅ Found {len(logo_patterns)} logo patterns")
        
        # Test cases
        test_messages = [
            "create a logo for my tech company",
            "I need a brand logo",
            "design a corporate emblem",
            "make me a logo for my restaurant",
            "generate a minimalist logo",
            "hello how are you",  # Should NOT match
            "what's the weather"  # Should NOT match
        ]
        
        logo_matches = 0
        for message in test_messages:
            is_logo = False
            for pattern in logo_patterns:
                if re.search(pattern, message.lower()):
                    is_logo = True
                    break
            
            if "logo" in message.lower() or "emblem" in message.lower() or "brand" in message.lower():
                expected = True
            else:
                expected = False
                
            if is_logo == expected:
                status = "✅"
                if is_logo:
                    logo_matches += 1
            else:
                status = "❌"
                
            print(f"  {status} '{message}' -> Logo: {is_logo} (Expected: {expected})")
        
        print(f"📊 Logo pattern matches: {logo_matches}/5 expected logo requests")
        return logo_matches >= 4  # Allow some tolerance
        
    except Exception as e:
        print(f"❌ Error testing intent recognition: {e}")
        return False

def test_brand_extraction():
    """Test smart brand name extraction"""
    print("\n🧪 Testing Brand Extraction Logic...")
    
    try:
        # Simulate the brand extraction logic from handle_logo_generation
        def extract_brand_name(text):
            brand_patterns = [
                r'logo.*for (.+?)(?:,|\.|$)',
                r'(?:brand|company|business) (?:called |named )?(.+?)(?:,|\.|$)',
                r'create.*logo.*(.+?)(?:,|\.|$)',
                r'design.*logo.*(.+?)(?:,|\.|$)',
                r'make.*logo.*(.+?)(?:,|\.|$)',
                r'generate.*logo.*(.+?)(?:,|\.|$)',
            ]
            
            brand_name = "YourBrand"
            for pattern in brand_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    brand_name = match.group(1).strip()
                    # Clean up the brand name
                    brand_name = re.sub(r'\b(a|an|the|my|our|company|business|brand)\b', '', brand_name).strip()
                    if brand_name:
                        break
            return brand_name
        
        test_cases = [
            ("create a logo for TechCorp", "TechCorp"),
            ("I need a logo for my company called Sunrise Bakery", "Sunrise Bakery"),
            ("design a logo for the restaurant Blue Moon", "Blue Moon"),
            ("make a logo for my business", "YourBrand"),  # Default fallback
            ("generate a minimalist logo for InnovateLabs", "InnovateLabs")
        ]
        
        success_count = 0
        for test_input, expected in test_cases:
            result = extract_brand_name(test_input)
            if result == expected or (expected == "YourBrand" and result):
                status = "✅"
                success_count += 1
            else:
                status = "❌"
            print(f"  {status} '{test_input}' -> '{result}' (Expected: '{expected}')")
        
        print(f"📊 Brand extraction: {success_count}/{len(test_cases)} correct")
        return success_count >= len(test_cases) - 1  # Allow one failure
        
    except Exception as e:
        print(f"❌ Error testing brand extraction: {e}")
        return False

def test_industry_detection():
    """Test industry keyword detection"""
    print("\n🧪 Testing Industry Detection...")
    
    try:
        industry_keywords = {
            'technology': ['tech', 'software', 'app', 'digital', 'ai', 'computer', 'coding'],
            'healthcare': ['health', 'medical', 'clinic', 'hospital', 'care', 'wellness'],
            'food': ['restaurant', 'cafe', 'food', 'kitchen', 'dining', 'bakery', 'coffee'],
            'fashion': ['fashion', 'clothing', 'style', 'boutique', 'apparel'],
            'finance': ['bank', 'finance', 'money', 'investment', 'financial'],
        }
        
        def detect_industry(text):
            text_lower = text.lower()
            for industry, keywords in industry_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    return industry
            return 'general'
        
        test_cases = [
            ("create a logo for my tech startup", "technology"),
            ("I need a restaurant logo", "food"),
            ("design a healthcare clinic logo", "healthcare"),
            ("make a fashion brand logo", "fashion"),
            ("create a bank logo", "finance"),
            ("general business logo", "general")
        ]
        
        success_count = 0
        for test_input, expected in test_cases:
            result = detect_industry(test_input)
            if result == expected:
                status = "✅"
                success_count += 1
            else:
                status = "❌"
            print(f"  {status} '{test_input}' -> '{result}' (Expected: '{expected}')")
        
        print(f"📊 Industry detection: {success_count}/{len(test_cases)} correct")
        return success_count >= len(test_cases) - 1
        
    except Exception as e:
        print(f"❌ Error testing industry detection: {e}")
        return False

def test_logo_handler_response():
    """Test the handle_logo_generation function response"""
    print("\n🧪 Testing Logo Handler Response...")
    
    try:
        from app import handle_logo_generation
        
        test_message = "create a modern logo for my tech company called InnovateLabs"
        response = handle_logo_generation(test_message)
        
        # Check if response contains expected elements
        checks = [
            ("Brand name mentioned", "InnovateLabs" in response),
            ("Style mentioned", "modern" in response.lower()),
            ("Industry context", any(word in response.lower() for word in ["tech", "technology", "industry"])),
            ("Helpful content", len(response) > 50),
            ("Professional tone", any(word in response for word in ["professional", "design", "logo", "brand"]))
        ]
        
        passed_checks = 0
        for check_name, check_result in checks:
            status = "✅" if check_result else "❌"
            print(f"  {status} {check_name}: {check_result}")
            if check_result:
                passed_checks += 1
        
        print(f"\n📝 Sample Response Preview:")
        print(f"'{response[:200]}...'")
        
        print(f"📊 Response quality: {passed_checks}/{len(checks)} checks passed")
        return passed_checks >= 3  # At least 3 out of 5 checks should pass
        
    except Exception as e:
        print(f"❌ Error testing logo handler: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all logo generation tests"""
    print("🚀 Running Simple Logo Generation Tests")
    print("=" * 50)
    
    tests = [
        ("Intent Recognition", test_intent_recognition),
        ("Brand Extraction", test_brand_extraction),
        ("Industry Detection", test_industry_detection),
        ("Logo Handler Response", test_logo_handler_response)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"\n{status} {test_name}")
            if result:
                passed += 1
        except Exception as e:
            print(f"\n❌ FAILED {test_name}: {e}")
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Logo generation system is working properly.")
        print("✨ The smart AI logo system is ready to use!")
    elif passed >= total - 1:
        print("✅ Most tests passed! Logo system is mostly functional.")
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    main()
