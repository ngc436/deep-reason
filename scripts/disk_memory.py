#!/usr/bin/env python3

import psutil
import sys
import os
import subprocess
from datetime import datetime

def get_docker_disk_usage():
    """Get Docker disk usage information."""
    try:
        # Get Docker disk usage using docker system df
        result = subprocess.run(['docker', 'system', 'df', '-v'], 
                              capture_output=True, text=True, check=True)
        print("\nDocker Disk Usage:")
        print("-" * 80)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error getting Docker disk usage: {e}", file=sys.stderr)
    except FileNotFoundError:
        print("Docker command not found. Make sure Docker is installed and in PATH.", file=sys.stderr)

def get_disk_usage():
    """Get disk usage information for all mounted partitions."""
    partitions = psutil.disk_partitions()
    
    print(f"\nSystem Disk Usage Information - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    print(f"{'Mount Point':<20} {'Total':>10} {'Used':>10} {'Free':>10} {'Usage %':>8}")
    print("-" * 80)
    
    for partition in partitions:
        try:
            usage = psutil.disk_usage(partition.mountpoint)
            total_gb = usage.total / (1024**3)
            used_gb = usage.used / (1024**3)
            free_gb = usage.free / (1024**3)
            
            print(f"{partition.mountpoint:<20} "
                  f"{total_gb:>10.1f}GB "
                  f"{used_gb:>10.1f}GB "
                  f"{free_gb:>10.1f}GB "
                  f"{usage.percent:>7.1f}%")
            
            # Check if we're close to quota limit (90% or more used)
            if usage.percent >= 90:
                print(f"WARNING: {partition.mountpoint} is at {usage.percent}% capacity!")
        except (PermissionError, FileNotFoundError):
            # Skip partitions that can't be accessed
            continue
        except OSError as e:
            if e.errno == 122:  # Disk quota exceeded
                print(f"ERROR: Disk quota exceeded on {partition.mountpoint}")
            else:
                print(f"Error accessing {partition.mountpoint}: {str(e)}")

def main():
    try:
        get_disk_usage()
        get_docker_disk_usage()
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 