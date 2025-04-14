#!/usr/bin/env python3
"""
Pokemon Red with Ollama deepseek-r1 Runner

This script provides an easy way to run the Ollama-powered Pokemon Red RL agent.
"""

import os
import subprocess
import sys

# Check if running in conda environment and activate 'main' if not already active
def ensure_conda_env():
    """Ensure we're running in the conda 'main' environment."""
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    print(f"Current conda environment: {conda_env}")
    
    if conda_env != 'main':
        print("Attempting to activate conda 'main' environment...")
        
        # Determine the conda executable
        if sys.platform.startswith('win'):
            conda_executable = 'conda.bat'
        else:
            conda_executable = 'conda'
        
        # Create a command to run this script with the 'main' environment
        script_path = os.path.abspath(__file__)
        
        # Get conda env paths
        try:
            env_result = subprocess.run([conda_executable, 'info', '--envs'], 
                                      stdout=subprocess.PIPE, 
                                      stderr=subprocess.PIPE,
                                      text=True)
            print(f"Available conda environments:\n{env_result.stdout}")
        except Exception as e:
            print(f"Error getting conda environments: {e}")
        
        # Try to launch the script in the conda environment
        print("\nPlease run this script using:")
        print(f"conda activate main && python {script_path}")
        
        # Exit this instance
        sys.exit(0)

# Call this at the beginning
ensure_conda_env()

# Diagnostic information
print("=== Diagnostic Information ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Script location: {os.path.abspath(__file__)}")

# Check for required Python packages
try:
    import requests
    print("✓ requests package is installed")
except ImportError:
    print("✗ requests package is missing")

try:
    import numpy
    print("✓ numpy package is installed")
except ImportError:
    print("✗ numpy package is missing")

try:
    import stable_baselines3
    print("✓ stable_baselines3 package is installed")
except ImportError:
    print("✗ stable_baselines3 package is missing")

# Check Ollama
print("\nChecking Ollama installation...")

def check_ollama_installed():
    """Check if Ollama is installed and accessible."""
    try:
        subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except FileNotFoundError:
        return False

def check_deepseek_model():
    """Check if the deepseek-r1 model is available."""
    print("Checking for deepseek-r1 model...")
    try:
        result = subprocess.run(["ollama", "list"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            print(f"Error running 'ollama list': {result.stderr}")
            return False
        
        if "deepseek-r1" in result.stdout:
            print("✓ deepseek-r1 model is available")
            # Get model details
            model_info = subprocess.run(["ollama", "show", "deepseek-r1"], 
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if model_info.returncode == 0:
                print(f"Model details: {model_info.stdout.strip()}")
            return True
        else:
            print("✗ deepseek-r1 model is not available")
            print(f"Available models: {result.stdout.strip()}")
            return False
    except Exception as e:
        print(f"Error checking for deepseek-r1 model: {e}")
        return False

def pull_deepseek_model():
    """Pull the deepseek-r1 model if not already available."""
    print("Downloading deepseek-r1 model (this may take a while)...")
    subprocess.run(["ollama", "pull", "deepseek-r1"], stdout=sys.stdout, stderr=sys.stderr)

def install_dependencies():
    """Install required Python dependencies."""
    dependencies = [
        "requests", 
        "numpy", 
        "stable-baselines3", 
        "scikit-image",
        "gym",
        "tensorboard",
        "pygame",
        "pillow"
    ]
    for dep in dependencies:
        try:
            __import__(dep.replace("-", "_"))
            print(f"✓ {dep} is already installed")
        except ImportError:
            print(f"Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"  {dep} installed successfully")

def main():
    """Main function to run the Ollama Pokemon agent."""
    # Check requirements
    if not check_ollama_installed():
        print("Ollama is not installed. Please install Ollama from https://ollama.ai/")
        print("After installing, run this script again.")
        return

    # Check and pull deepseek-r1 model if needed
    if not check_deepseek_model():
        print("The deepseek-r1 model is not available.")
        answer = input("Would you like to download it now? (y/n): ")
        if answer.lower() == "y":
            pull_deepseek_model()
        else:
            print("The deepseek-r1 model is required to run this program.")
            return

    # Install Python dependencies
    install_dependencies()

    # Ask user what they want to do
    print("\nPokemon Red with Ollama deepseek-r1")
    print("-----------------------------------")
    print("1. Train a new model with deepseek-r1 guidance")
    print("2. Play with an existing model and deepseek-r1 assistance")
    print("3. Both train and play")
    
    choice = input("\nEnter your choice (1-3): ")
    
    # Run the agent with the selected mode
    if choice == "1":
        subprocess.run([sys.executable, "ollama_pokemon_agent.py", "--train"],
                      cwd=os.path.dirname(os.path.abspath(__file__)))
    elif choice == "2":
        subprocess.run([sys.executable, "ollama_pokemon_agent.py", "--play"],
                      cwd=os.path.dirname(os.path.abspath(__file__)))
    else:
        subprocess.run([sys.executable, "ollama_pokemon_agent.py"],
                      cwd=os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    main() 