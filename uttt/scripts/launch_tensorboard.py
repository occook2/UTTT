#!/usr/bin/env python3
"""
Script to launch TensorBoard for viewing training logs.
"""
import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Launch TensorBoard for UTTT training logs")
    parser.add_argument('--mode', choices=['individual', 'compare', 'all'], default='compare',
                       help='Viewing mode: individual (single run), compare (all runs), all (hierarchical)')
    parser.add_argument('--run', type=str, help='Specific run directory for individual mode')
    parser.add_argument('--port', type=int, default=6006, help='TensorBoard port')
    
    args = parser.parse_args()
    
    if args.mode == 'individual':
        if not args.run:
            print("Error: --run is required for individual mode")
            sys.exit(1)
        
        log_dir = os.path.join('runs', args.run, 'tensorboard')
        if not os.path.exists(log_dir):
            print(f"Error: TensorBoard logs not found at {log_dir}")
            sys.exit(1)
        
        print(f"Launching TensorBoard for individual run: {args.run}")
        print(f"URL: http://localhost:{args.port}")
        
    elif args.mode == 'compare':
        log_dir = 'tensorboard_logs'
        if not os.path.exists(log_dir):
            print(f"Error: Shared TensorBoard logs not found at {log_dir}")
            print("Make sure you have completed at least one training run with TensorBoard enabled.")
            sys.exit(1)
        
        print("Launching TensorBoard for comparing multiple runs")
        print(f"URL: http://localhost:{args.port}")
        
    elif args.mode == 'all':
        log_dir = 'runs'
        if not os.path.exists(log_dir):
            print(f"Error: Runs directory not found at {log_dir}")
            sys.exit(1)
        
        print("Launching TensorBoard with hierarchical view of all runs")
        print(f"URL: http://localhost:{args.port}")
    
    # Launch TensorBoard
    try:
        cmd = ['tensorboard', '--logdir', log_dir, '--port', str(args.port)]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nTensorBoard stopped.")
    except FileNotFoundError:
        print("Error: TensorBoard not found. Install it with: pip install tensorboard")
        sys.exit(1)

if __name__ == "__main__":
    main()