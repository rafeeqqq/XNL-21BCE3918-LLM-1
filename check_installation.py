#!/usr/bin/env python3
"""
Installation Check Script for AI-Powered FinTech Platform

This script checks if all required dependencies are installed and if the services
are properly configured.
"""

import importlib
import os
import sys
import subprocess
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
        print(f"❌ Python version {python_version.major}.{python_version.minor} is not supported. Please use Python 3.9+")
        return False
    print(f"✅ Python version {python_version.major}.{python_version.minor}.{python_version.micro} is compatible")
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\nChecking dependencies...")
    
    # Read requirements.txt
    try:
        with open('requirements.txt', 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False
    
    # Check each dependency
    all_installed = True
    installed_packages = []
    missing_packages = []
    version_mismatch_packages = []
    
    for requirement in requirements:
        try:
            pkg_resources.require(requirement)
            installed_packages.append(requirement)
        except DistributionNotFound:
            missing_packages.append(requirement)
            all_installed = False
        except VersionConflict as e:
            version_mismatch_packages.append((requirement, str(e.dist)))
            all_installed = False
    
    # Print results
    print(f"Total dependencies: {len(requirements)}")
    print(f"Installed: {len(installed_packages)}")
    
    if missing_packages:
        print(f"\n❌ Missing packages ({len(missing_packages)}):")
        for pkg in missing_packages:
            print(f"  - {pkg}")
    
    if version_mismatch_packages:
        print(f"\n⚠️ Version mismatch ({len(version_mismatch_packages)}):")
        for pkg, installed in version_mismatch_packages:
            print(f"  - Required: {pkg}, Installed: {installed}")
    
    if all_installed:
        print("✅ All dependencies are installed correctly")
    else:
        print("\nTo install missing dependencies, run:")
        print("pip install -r requirements.txt")
    
    return all_installed

def check_services():
    """Check if all services are properly configured"""
    print("\nChecking services...")
    
    services = [
        {"name": "Data Ingestion Service", "path": "services/data_ingestion/main.py"},
        {"name": "LLM Service", "path": "services/llm_service/main.py"},
        {"name": "Trading Engine", "path": "services/trading_engine/main.py"}
    ]
    
    all_services_ok = True
    
    for service in services:
        if os.path.isfile(service["path"]):
            print(f"✅ {service['name']} found at {service['path']}")
        else:
            print(f"❌ {service['name']} not found at {service['path']}")
            all_services_ok = False
    
    if all_services_ok:
        print("\nAll services are properly configured")
    else:
        print("\nSome services are missing. Please check the project structure.")
    
    return all_services_ok

def check_orchestrator():
    """Check if the service orchestrator is properly configured"""
    print("\nChecking service orchestrator...")
    
    if os.path.isfile("start_services.py"):
        if os.access("start_services.py", os.X_OK):
            print("✅ Service orchestrator is executable")
        else:
            print("⚠️ Service orchestrator is not executable. Run: chmod +x start_services.py")
        return True
    else:
        print("❌ Service orchestrator (start_services.py) not found")
        return False

def main():
    """Main function"""
    print("=" * 80)
    print("AI-POWERED FINTECH PLATFORM - INSTALLATION CHECK")
    print("=" * 80)
    
    python_ok = check_python_version()
    dependencies_ok = check_dependencies()
    services_ok = check_services()
    orchestrator_ok = check_orchestrator()
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Python version: {'✅ OK' if python_ok else '❌ NOT OK'}")
    print(f"Dependencies: {'✅ OK' if dependencies_ok else '❌ NOT OK'}")
    print(f"Services: {'✅ OK' if services_ok else '❌ NOT OK'}")
    print(f"Service orchestrator: {'✅ OK' if orchestrator_ok else '❌ NOT OK'}")
    
    if python_ok and dependencies_ok and services_ok and orchestrator_ok:
        print("\n✅ All checks passed! You can start the platform with:")
        print("./start_services.py")
    else:
        print("\n⚠️ Some checks failed. Please fix the issues before starting the platform.")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
