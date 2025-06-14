import csv
import time

from simulation import run_simulation
import random
random.seed()
n = 3  # number of runs
all_results = [n]

# change these numbers according to scenarios
agv_number = 2
charge_threshold = 45
agv_speed = 2
asrs_mean = 60
asrs_std = 45


def load_seeds_from_csv():
    seeds = []
    with open('seeds.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            seeds.append(int(row[0]))
    return seeds

# Usage
seed_list = load_seeds_from_csv()

generic_results_list = []
results_on_list = []
results_off_list = []
start_time = time.time()
for i in range(n):
    progress = (i + 1) / n * 100
    bar_length = 35
    filled_length = int(bar_length * (i + 1) // n)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    elapsed_time = time.time() - start_time
    print(f'\rProgress: |{bar}| {progress:.1f}% ({i + 1}/{n}) current seed: ' + str(seed_list[i]), end='', flush=True)

    generic_results, results_on = run_simulation(innovation=True, seed=seed_list[i], agv_number=agv_number,
                                                 charge_threshold=charge_threshold, agv_speed=agv_speed,
                                                 asrs_mean=asrs_mean, asrs_std=asrs_std)
    _, results_off = run_simulation(innovation=False, seed=seed_list[i], agv_number=agv_number,
                                    charge_threshold=charge_threshold, agv_speed=agv_speed,
                                    asrs_mean=asrs_mean, asrs_std=asrs_std)

    generic_results_list.append(generic_results)
    results_on_list.append(results_on)
    results_off_list.append(results_off)

total_time = time.time() - start_time
print(f"\n===== {n} SIMULATIONS DONE in {total_time:.1f} SECONDS =====")
print(f"\nNote: All results below are average values across the simulations.")
print()
average_generic = {}
average_on = {}
average_off = {}

for key in generic_results_list[0]:
    average_generic[key] = sum(generic_result[key] for generic_result in generic_results_list) / n

for key in results_on_list[0]:
    average_on[key] = sum(result[key] for result in results_on_list) / n

for key in results_off_list[0]:
    average_off[key] = sum(result[key] for result in results_off_list) / n

for key, value in average_generic.items():
    if isinstance(value, (int, float)):
        print(f"{key}: {value:.1f}")
    else:
        print(f"{key}: {value}")

print("\n===== NON-INNOVATION RESULTS =====")
for key, value in average_off.items():
    if isinstance(value, (int, float)):
        print(f"{key}: {value:.1f}")
    else:
        print(f"{key}: {value}")

print("\n===== INNOVATION RESULTS =====")
for key, value in average_on.items():
    if isinstance(value, (int, float)):
        print(f"{key}: {value:.1f}")
    else:
        print(f"{key}: {value}")
