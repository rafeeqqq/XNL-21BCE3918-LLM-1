#!/usr/bin/env python3
"""
Service Orchestrator for AI-Powered FinTech Platform

This script starts all the necessary services in the correct order:
1. Data Ingestion Service
2. LLM Service
3. Trading Engine
4. API Gateway (if available)

It also monitors the services and provides a simple dashboard of their status.
"""

import os
import sys
import time
import signal
import subprocess
import threading
import requests
from datetime import datetime
import argparse

# Service configurations
SERVICES = [
    {
        "name": "Data Ingestion Service",
        "directory": "services/data_ingestion",
        "command": "python main.py",
        "port": 8001,
        "health_endpoint": "/health",
        "process": None,
        "status": "Stopped",
        "startup_time": None
    },
    {
        "name": "LLM Service",
        "directory": "services/llm_service",
        "command": "python main.py",
        "port": 8002,
        "health_endpoint": "/health",
        "process": None,
        "status": "Stopped",
        "startup_time": None
    },
    {
        "name": "Trading Engine",
        "directory": "services/trading_engine",
        "command": "python main.py",
        "port": 8003,
        "health_endpoint": "/health",
        "process": None,
        "status": "Stopped",
        "startup_time": None
    },
    {
        "name": "API Gateway",
        "directory": "api",
        "command": "python main.py",
        "port": 8000,
        "health_endpoint": "/health",
        "process": None,
        "status": "Stopped",
        "startup_time": None
    }
]

# Global variables
running = True
base_dir = os.path.dirname(os.path.abspath(__file__))

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_dashboard():
    """Print a dashboard of service statuses"""
    clear_screen()
    print("\n" + "=" * 80)
    print(f"{'AI-POWERED FINTECH PLATFORM - SERVICE DASHBOARD':^80}")
    print("=" * 80)
    print(f"{'Service Name':<30} {'Status':<15} {'Port':<10} {'Uptime':<20}")
    print("-" * 80)
    
    for service in SERVICES:
        # Calculate uptime if service is running
        uptime = ""
        if service["startup_time"] is not None:
            uptime_seconds = (datetime.now() - service["startup_time"]).total_seconds()
            minutes, seconds = divmod(int(uptime_seconds), 60)
            hours, minutes = divmod(minutes, 60)
            uptime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        print(f"{service['name']:<30} {service['status']:<15} {service['port']:<10} {uptime:<20}")
    
    print("-" * 80)
    print("Commands: [r] Restart Service, [s] Stop Service, [t] Run Tests, [q] Quit")
    print("=" * 80)

def check_service_health(service):
    """Check if a service is healthy by pinging its health endpoint"""
    try:
        response = requests.get(f"http://localhost:{service['port']}{service['health_endpoint']}", timeout=2)
        return response.status_code == 200
    except:
        return False

def monitor_services():
    """Monitor the health of all services"""
    global running
    
    while running:
        for service in SERVICES:
            if service["process"] is not None and service["process"].poll() is None:
                # Process is running, check health
                if check_service_health(service):
                    service["status"] = "Running"
                else:
                    service["status"] = "Unhealthy"
            elif service["process"] is not None:
                # Process has terminated
                service["status"] = "Crashed"
                service["startup_time"] = None
            
        # Update dashboard
        print_dashboard()
        
        # Wait before checking again
        time.sleep(5)

def start_service(service_index):
    """Start a specific service"""
    service = SERVICES[service_index]
    
    # Check if service is already running
    if service["process"] is not None and service["process"].poll() is None:
        print(f"{service['name']} is already running.")
        return
    
    # Start the service
    try:
        service_dir = os.path.join(base_dir, service["directory"])
        
        # Check if directory exists
        if not os.path.isdir(service_dir):
            print(f"Error: Directory {service_dir} does not exist.")
            return
        
        # Check if main.py exists
        if not os.path.isfile(os.path.join(service_dir, "main.py")):
            print(f"Error: main.py not found in {service_dir}.")
            return
        
        # Start the process
        process = subprocess.Popen(
            service["command"].split(),
            cwd=service_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        service["process"] = process
        service["status"] = "Starting"
        service["startup_time"] = datetime.now()
        
        print(f"Started {service['name']} (PID: {process.pid})")
        
        # Wait for service to become healthy
        max_attempts = 10
        for attempt in range(max_attempts):
            time.sleep(2)
            if check_service_health(service):
                service["status"] = "Running"
                print(f"{service['name']} is now healthy.")
                break
            print(f"Waiting for {service['name']} to become healthy... ({attempt+1}/{max_attempts})")
        else:
            print(f"Warning: {service['name']} did not become healthy within the timeout period.")
    
    except Exception as e:
        print(f"Error starting {service['name']}: {e}")

def stop_service(service_index):
    """Stop a specific service"""
    service = SERVICES[service_index]
    
    if service["process"] is None or service["process"].poll() is not None:
        print(f"{service['name']} is not running.")
        return
    
    try:
        # Try to terminate gracefully first
        service["process"].terminate()
        
        # Wait for up to 5 seconds for the process to terminate
        for _ in range(5):
            if service["process"].poll() is not None:
                break
            time.sleep(1)
        
        # If still running, kill it
        if service["process"].poll() is None:
            service["process"].kill()
            service["process"].wait()
        
        service["status"] = "Stopped"
        service["startup_time"] = None
        print(f"Stopped {service['name']}")
    
    except Exception as e:
        print(f"Error stopping {service['name']}: {e}")

def start_all_services():
    """Start all services in the correct order"""
    for i in range(len(SERVICES)):
        start_service(i)
        time.sleep(2)  # Give some time between service starts

def stop_all_services():
    """Stop all services in reverse order"""
    for i in range(len(SERVICES) - 1, -1, -1):
        stop_service(i)

def run_tests():
    """Run the test script"""
    print("Running tests...")
    test_script = os.path.join(base_dir, "test_services.py")
    
    if not os.path.isfile(test_script):
        print(f"Error: Test script {test_script} not found.")
        return
    
    try:
        subprocess.run([sys.executable, test_script], check=True)
        print("Tests completed.")
    except subprocess.CalledProcessError:
        print("Tests failed. Check the output for details.")
    
    input("Press Enter to continue...")

def handle_user_input():
    """Handle user input for controlling services"""
    global running
    
    while running:
        command = input().lower()
        
        if command == 'q':
            running = False
            stop_all_services()
            print("Shutting down all services...")
            break
        
        elif command == 't':
            run_tests()
        
        elif command.startswith('r'):
            # Restart a service
            try:
                service_index = int(command[1:]) - 1
                if 0 <= service_index < len(SERVICES):
                    stop_service(service_index)
                    time.sleep(1)
                    start_service(service_index)
                else:
                    print(f"Invalid service index. Please enter a number between 1 and {len(SERVICES)}.")
            except ValueError:
                print("Invalid command. Use 'r<number>' to restart a service (e.g., 'r1').")
        
        elif command.startswith('s'):
            # Stop a service
            try:
                service_index = int(command[1:]) - 1
                if 0 <= service_index < len(SERVICES):
                    stop_service(service_index)
                else:
                    print(f"Invalid service index. Please enter a number between 1 and {len(SERVICES)}.")
            except ValueError:
                print("Invalid command. Use 's<number>' to stop a service (e.g., 's1').")

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully shut down"""
    global running
    print("\nShutting down...")
    running = False
    stop_all_services()
    sys.exit(0)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="AI-Powered FinTech Platform Service Orchestrator")
    parser.add_argument("--test-only", action="store_true", help="Run tests without starting services")
    args = parser.parse_args()
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    if args.test_only:
        run_tests()
        return
    
    # Start all services
    start_all_services()
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_services)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Handle user input
    handle_user_input()

if __name__ == "__main__":
    main()
