#!/usr/bin/env python3
"""
Comprehensive Test Runner for Horizon AI
Runs all tests with detailed reporting and coverage analysis
"""

import sys
import os
import subprocess
import argparse
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(command)}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, check=True, 
                              capture_output=False, text=True)
        duration = time.time() - start_time
        print(f"\n✅ {description} completed successfully in {duration:.2f}s")
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"\n❌ {description} failed after {duration:.2f}s")
        print(f"Exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found: {command[0]}")
        return False

def run_unittest_tests():
    """Run unittest-based tests"""
    print("\n🧪 Running unittest-based tests...")
    
    test_files = [
        'tests/test_comprehensive.py',
        'tests/test_personality_blending.py',
        'tests/test_api_endpoints.py'
    ]
    
    results = []
    for test_file in test_files:
        if os.path.exists(test_file):
            success = run_command(
                ['python3', test_file], 
                f"Running {test_file}"
            )
            results.append((test_file, success))
        else:
            print(f"⚠️ Test file not found: {test_file}")
            results.append((test_file, False))
    
    return results

def run_pytest_tests():
    """Run pytest-based tests with coverage"""
    return run_command(
        ['python3', '-m', 'pytest', 'tests/', '--cov=app', '--cov=utils', 
         '--cov-report=html', '--cov-report=term', '--verbose'],
        "Running pytest with coverage analysis"
    )

def check_test_dependencies():
    """Check if test dependencies are installed"""
    print("🔍 Checking test dependencies...")
    
    required_packages = ['pytest', 'coverage', 'mock']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def run_linting():
    """Run code linting and style checks"""
    linting_commands = [
        (['python3', '-m', 'flake8', 'app.py', 'utils/', '--max-line-length=100'], 
         "Running flake8 linting"),
        (['python3', '-m', 'pylint', 'app.py'], 
         "Running pylint analysis"),
    ]
    
    results = []
    for command, description in linting_commands:
        try:
            success = run_command(command, description)
            results.append((description, success))
        except Exception:
            print(f"⚠️ Skipping {description} - tool not available")
            results.append((description, None))
    
    return results

def run_security_checks():
    """Run security checks"""
    security_commands = [
        (['python3', '-m', 'bandit', '-r', 'app.py'], 
         "Running bandit security analysis"),
        (['python3', '-m', 'safety', 'check'], 
         "Running safety vulnerability check"),
    ]
    
    results = []
    for command, description in security_commands:
        try:
            success = run_command(command, description)
            results.append((description, success))
        except Exception:
            print(f"⚠️ Skipping {description} - tool not available")
            results.append((description, None))
    
    return results

def generate_test_report(unittest_results, pytest_success, linting_results, security_results):
    """Generate comprehensive test report"""
    print("\n" + "="*80)
    print("📊 HORIZON AI COMPREHENSIVE TEST REPORT")
    print("="*80)
    
    # Summary statistics
    total_unittest_tests = len(unittest_results)
    passed_unittest_tests = sum(1 for _, success in unittest_results if success)
    
    print(f"\n🧪 UNITTEST RESULTS:")
    print(f"   Total test suites: {total_unittest_tests}")
    print(f"   Passed: {passed_unittest_tests}")
    print(f"   Failed: {total_unittest_tests - passed_unittest_tests}")
    
    for test_file, success in unittest_results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {status}: {test_file}")
    
    print(f"\n🔬 PYTEST RESULTS:")
    status = "✅ PASS" if pytest_success else "❌ FAIL"
    print(f"   {status}: Pytest with coverage")
    
    if linting_results:
        print(f"\n🔍 CODE QUALITY:")
        for description, success in linting_results:
            if success is None:
                status = "⏭️ SKIP"
            elif success:
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            print(f"   {status}: {description}")
    
    if security_results:
        print(f"\n🔒 SECURITY CHECKS:")
        for description, success in security_results:
            if success is None:
                status = "⏭️ SKIP"
            elif success:
                status = "✅ PASS"
            else:
                status = "❌ FAIL"
            print(f"   {status}: {description}")
    
    # Overall status
    overall_success = (
        passed_unittest_tests == total_unittest_tests and
        pytest_success
    )
    
    print(f"\n🏆 OVERALL STATUS: {'✅ SUCCESS' if overall_success else '❌ SOME FAILURES'}")
    
    # Coverage report location
    coverage_html = PROJECT_ROOT / "htmlcov" / "index.html"
    if coverage_html.exists():
        print(f"\n📈 Coverage report: file://{coverage_html}")
    
    print("\n" + "="*80)
    
    return overall_success

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Horizon AI Test Runner")
    parser.add_argument("--quick", action="store_true", 
                       help="Run only quick tests (skip slow integration tests)")
    parser.add_argument("--coverage", action="store_true", 
                       help="Run with coverage analysis")
    parser.add_argument("--lint", action="store_true", 
                       help="Run linting and code quality checks")
    parser.add_argument("--security", action="store_true", 
                       help="Run security checks")
    parser.add_argument("--all", action="store_true", 
                       help="Run all tests and checks")
    
    args = parser.parse_args()
    
    if not any([args.quick, args.coverage, args.lint, args.security, args.all]):
        # Default: run core tests
        args.coverage = True
    
    if args.all:
        args.coverage = args.lint = args.security = True
    
    print("🚀 HORIZON AI COMPREHENSIVE TEST SUITE")
    print("="*50)
    
    start_time = time.time()
    
    # Check dependencies
    if not check_test_dependencies():
        print("❌ Missing test dependencies. Please install requirements.txt")
        return 1
    
    # Run tests
    unittest_results = []
    pytest_success = False
    linting_results = []
    security_results = []
    
    # Always run unittest tests
    unittest_results = run_unittest_tests()
    
    # Run pytest with coverage if requested
    if args.coverage:
        pytest_success = run_pytest_tests()
    
    # Run linting if requested
    if args.lint:
        linting_results = run_linting()
    
    # Run security checks if requested
    if args.security:
        security_results = run_security_checks()
    
    # Generate report
    overall_success = generate_test_report(
        unittest_results, pytest_success, linting_results, security_results
    )
    
    total_time = time.time() - start_time
    print(f"\n⏱️ Total execution time: {total_time:.2f} seconds")
    
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())