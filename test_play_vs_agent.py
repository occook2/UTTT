#!/usr/bin/env python3
"""
Test script to verify the play_vs_agent functionality.
"""

if __name__ == "__main__":
    import sys
    import os
    
    # Add the current directory to path to ensure imports work
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from uttt.scripts.play_vs_agent import main
    main()