"""
Script to run Differential Privacy Federated Learning
"""

import subprocess
import time
import os
import sys
from pathlib import Path

def run_dp_federated_learning():
    """Run the complete DP federated learning system"""
    
    print("="*80)
    print("DIFFERENTIAL PRIVACY FEDERATED LEARNING SYSTEM")
    print("="*80)
    
    # Check if required files exist
    required_files = [
        "dp_utils.py",
        "dp_server.py", 
        "dp_client_1.py",
        "dp_client_2.py",
        "dp_client_3.py",
        "model.py",
        "utils.py"
    ]
    
    print("Checking required files...")
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files found!")
    
    # Create necessary directories
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("scalers", exist_ok=True)
    
    print("\n" + "="*50)
    print("INSTRUCTIONS:")
    print("="*50)
    print("1. First, run the server: python dp_server.py")
    print("2. Then run each client in separate terminals:")
    print("   - Terminal 2: python dp_client_1.py")
    print("   - Terminal 3: python dp_client_2.py") 
    print("   - Terminal 4: python dp_client_3.py")
    print("\nOR use the automated script below...")
    print("="*50)
    
    # Ask user for preference
    choice = input("\nDo you want to run automatically? (y/n): ").lower().strip()
    
    if choice == 'y':
        run_automated()
    else:
        print("\nManual execution:")
        print("Open 4 separate terminals and run the commands above.")
        print("The server will wait for all 3 clients to connect.")
    
    return True

def run_automated():
    """Run the system automatically using subprocesses"""
    
    print("\n🚀 Starting automated DP federated learning...")
    
    processes = []
    
    try:
        # Start server
        print("Starting DP server...")
        server_process = subprocess.Popen([
            sys.executable, "dp_server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(("Server", server_process))
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Start clients
        clients = ["dp_client_1.py", "dp_client_2.py", "dp_client_3.py"]
        
        for i, client in enumerate(clients, 1):
            print(f"Starting DP client {i}...")
            client_process = subprocess.Popen([
                sys.executable, client
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            processes.append((f"Client {i}", client_process))
            time.sleep(1)  # Small delay between clients
        
        print(f"\n✅ All processes started!")
        print("🔄 Training in progress...")
        print("📊 Monitor the output for privacy metrics...")
        
        # Wait for server to complete (it will finish when all rounds are done)
        server_process.wait()
        print("\n🎉 DP Federated Learning completed!")
        
        # Terminate any remaining client processes
        for name, process in processes[1:]:  # Skip server (already finished)
            if process.poll() is None:  # Still running
                process.terminate()
                print(f"Terminated {name}")
        
        print("\n📁 Check the 'saved_models' directory for DP models!")
        print("🔍 Look for files with '_dp_' prefix for differential privacy models.")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user!")
        # Terminate all processes
        for name, process in processes:
            if process.poll() is None:
                process.terminate()
                print(f"Terminated {name}")
    
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        # Terminate all processes
        for name, process in processes:
            if process.poll() is None:
                process.terminate()
                print(f"Terminated {name}")

def check_dp_results():
    """Check and display DP results"""
    
    print("\n" + "="*60)
    print("CHECKING DP RESULTS")
    print("="*60)
    
    # Check for saved models
    saved_models_dir = Path("saved_models")
    if saved_models_dir.exists():
        dp_models = list(saved_models_dir.glob("*dp*"))
        if dp_models:
            print("✅ DP Models found:")
            for model in dp_models:
                print(f"  📁 {model.name}")
        else:
            print("❌ No DP models found")
    else:
        print("❌ No saved_models directory found")
    
    # Check for privacy analysis
    privacy_files = list(saved_models_dir.glob("*privacy_analysis*")) if saved_models_dir.exists() else []
    if privacy_files:
        print(f"\n✅ Privacy analysis files found:")
        for file in privacy_files:
            print(f"  📊 {file.name}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    success = run_dp_federated_learning()
    
    if success:
        print("\n🎯 Next steps:")
        print("1. Run the federated learning system")
        print("2. Check results with: python -c \"from run_dp_federated_learning import check_dp_results; check_dp_results()\"")
        print("3. Test the DP model with your existing model_evaluator.py")
        
        # Ask if user wants to check results now
        if input("\nCheck for existing results now? (y/n): ").lower().strip() == 'y':
            check_dp_results()