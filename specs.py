#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 19:47:23 2024

@author: angel
"""
import platform
import psutil
import timeit
import multiprocessing
import subprocess

def get_system_info():
    system_info = {
        'System': platform.system(),
        'Node': platform.node(),
        'Release': platform.release(),
        'Version': platform.version(),
        'Machine': platform.machine(),
        'Processor': platform.processor(),
        'CPU Cores': psutil.cpu_count(logical=False),  # Physical cores
        'CPU Threads': psutil.cpu_count(logical=True),  # Logical cores (with hyperthreading)
        'CPU Frequency (GHz)': psutil.cpu_freq().current / 1000,  # GHz
    }
    return system_info

def get_cpu_load():
    return psutil.cpu_percent(percpu=True)

def run_external_script(script_name):
    try:
        subprocess.run(["python", script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")

if __name__ == "__main__":
    start_time = timeit.default_timer()

    system_info = get_system_info()
    cpu_cores = multiprocessing.cpu_count()

    print("Computer Specifications:")
    for key, value in system_info.items():
        print(f"{key}: {value}")

    print("\nNumber of CPU Cores: {}".format(cpu_cores))

    # Record CPU usage before running the external script
    initial_cpu_usage = get_cpu_load()

    # Run the external script (main.py in this case)
    run_external_script("main.py")

    # Record CPU usage after running the external script
    final_cpu_usage = get_cpu_load()

    # Determine the number of cores used
    used_cores = sum(1 for initial, final in zip(initial_cpu_usage, final_cpu_usage) if final > initial)

    print("\nNumber of Cores Used: {}".format(used_cores))

    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time
    print("\nElapsed Time: {:.2f} seconds".format(elapsed_time))
