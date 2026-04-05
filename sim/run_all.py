#!/usr/bin/env python3
"""
Run all five experiments in sequence.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

def main():
    total_start = time.time()
    
    print("\n" + "=" * 70)
    print("  A GUT-DRIVEN SEARCH FOR A GUT — FULL EXPERIMENTAL SUITE")
    print("=" * 70 + "\n")
    
    # Experiment 1
    print("\n" + "▓" * 70)
    from exp1_sign import main as exp1
    exp1()
    
    # Experiment 2
    print("\n" + "▓" * 70)
    from exp2_confinement import main as exp2
    exp2()
    
    # Experiment 3
    print("\n" + "▓" * 70)
    from exp3_2d import main as exp3
    exp3()
    
    # Experiment 4
    print("\n" + "▓" * 70)
    from exp4_conservation import main as exp4
    exp4()
    
    # Experiment 5
    print("\n" + "▓" * 70)
    from exp5_coupling_ratios import main as exp5
    exp5()
    
    elapsed = time.time() - total_start
    
    print("\n" + "=" * 70)
    print(f"  ALL EXPERIMENTS COMPLETE — {elapsed:.1f}s total")
    print("=" * 70)
    print("\n  Next: python sim/visualize_experiments.py  (generate plots)")


if __name__ == "__main__":
    main()
