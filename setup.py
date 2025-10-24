#!/usr/bin/env python3
"""
UTTT (Ultimate Tic-Tac-Toe) AlphaZero Setup Script

This script sets up the project environment and verifies the installation.
Run this after installing requirements to ensure everything is working correctly.
"""

import os
import sys
import subprocess
import yaml
import torch
import importlib.util
from pathlib import Path

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Found: {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('yaml', 'PyYAML'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),
        ('tqdm', 'tqdm'),
        ('tkinter', 'tkinter')
    ]
    
    all_good = True
    for package, name in required_packages:
        try:
            if package == 'yaml':
                import yaml
            elif package == 'tkinter':
                import tkinter as tk
            else:
                __import__(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - run: pip install {package}")
            all_good = False
    
    return all_good

def check_cuda():
    """Check CUDA availability for PyTorch"""
    print("\n🔍 Checking CUDA support...")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            print(f"✅ CUDA available: {device_count} device(s)")
            print(f"   Current device: {device_name}")
            return True
        else:
            print("⚠️  CUDA not available - will use CPU")
            print("   For GPU acceleration, install CUDA-enabled PyTorch:")
            print("   https://pytorch.org/get-started/locally/")
            return False
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False

def verify_project_structure():
    """Verify that the project structure is correct"""
    print("\n🔍 Verifying project structure...")
    
    required_dirs = [
        'uttt',
        'uttt/agents',
        'uttt/agents/az',
        'uttt/env',
        'uttt/mcts',
        'uttt/scripts',
        'uttt/tests'
    ]
    
    required_files = [
        'config.yaml',
        'uttt/__init__.py',
        'uttt/agents/az/train.py',
        'uttt/env/state.py',
        'uttt/scripts/play_vs_agent.py'
    ]
    
    project_root = Path(__file__).parent
    
    all_good = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - missing directory")
            all_good = False
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - missing file")
            all_good = False
    
    return all_good

def create_directories():
    """Create necessary directories for training"""
    print("\n🔧 Creating directories...")
    
    dirs_to_create = [
        'runs',
        'tensorboard_logs',
        'tournaments',
        'checkpoints'
    ]
    
    project_root = Path(__file__).parent
    
    for dir_name in dirs_to_create:
        dir_path = project_root / dir_name
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created {dir_name}/")
        else:
            print(f"✅ {dir_name}/ already exists")

def verify_config():
    """Verify that config.yaml is valid"""
    print("\n🔍 Verifying configuration...")
    
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for required sections
        required_sections = ['alphazero', 'mcts', 'training', 'model']
        for section in required_sections:
            if section in config:
                print(f"✅ {section} section")
            else:
                print(f"❌ {section} section missing")
                return False
        
        return True
    except Exception as e:
        print(f"❌ Error reading config.yaml: {e}")
        return False

def run_quick_test():
    """Run a quick test to ensure the core functionality works"""
    print("\n🧪 Running quick test...")
    
    try:
        # Test importing core modules
        sys.path.insert(0, '.')
        
        from uttt.env.state import UTTTState
        from uttt.agents.random import RandomAgent
        
        # Create a simple game state
        state = UTTTState()
        agent = RandomAgent()
        
        # Make a few moves
        for _ in range(3):
            if not state.is_terminal():
                valid_actions = state.get_valid_actions()
                if valid_actions:
                    action = agent.get_action(state)
                    state = state.make_move(action)
        
        print("✅ Core functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        return False

def display_next_steps():
    """Display next steps for the user"""
    print("\n" + "="*60)
    print("🎉 Setup Complete! Next Steps:")
    print("="*60)
    
    print("\n1. 🏃 Quick Start - Train a small model:")
    print("   python -m uttt.agents.az.train")
    
    print("\n2. 🎮 Play against an agent:")
    print("   python -m uttt.scripts.play_vs_agent")
    
    print("\n3. 📊 View training data:")
    print("   python -m uttt.scripts.view_training_data")
    
    print("\n4. 🏆 Run tournaments:")
    print("   python -m uttt.scripts.alphazero_tournament")
    
    print("\n5. 📈 Launch TensorBoard:")
    print("   python -m uttt.scripts.launch_tensorboard")
    
    print("\n6. ⚙️  Customize training:")
    print("   Edit config.yaml to adjust parameters")
    
    print("\n📚 Documentation:")
    print("   - README.md - Complete project overview")
    print("   - config.yaml - Configuration reference")
    
    print("\n💡 Tips:")
    print("   - Start with a small model for testing")
    print("   - Use CPU for initial testing if no CUDA")
    print("   - Check tensorboard_logs/ for training progress")
    
    print("\n🐛 Troubleshooting:")
    print("   - For CUDA issues: reinstall PyTorch with CUDA")
    print("   - For tkinter issues: install python3-tk")
    print("   - For import errors: run from project root")

def main():
    """Main setup function"""
    print("🚀 UTTT AlphaZero Setup")
    print("="*40)
    
    checks = [
        ("Python version", check_python_version),
        ("Dependencies", check_dependencies),
        ("CUDA support", check_cuda),
        ("Project structure", verify_project_structure),
        ("Configuration", verify_config),
        ("Core functionality", run_quick_test)
    ]
    
    all_passed = True
    for name, check_func in checks:
        if not check_func():
            all_passed = False
    
    # Always create directories even if some checks fail
    create_directories()
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All checks passed!")
        display_next_steps()
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        print("\n🔧 Common fixes:")
        print("   - Install missing packages: pip install -r requirements.txt")
        print("   - For CUDA: visit https://pytorch.org/get-started/locally/")
        print("   - For tkinter: install system package (python3-tk)")
    
    print("\n🎯 Ready to start training your AlphaZero agent!")
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)