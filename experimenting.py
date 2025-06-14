import time

from simulation import run_simulation
import statistics  # To compute standard deviation
from scipy.stats import ttest_rel
import numpy as np

import random
random.seed()
n = 100  # number of runs
all_results = [n]

# change these numbers according to scenarios
agv_number = 4
charge_threshold = 90
agv_speed = 2
asrs_mean = 60
asrs_std = 45

seed_list=[958979, 714759, 755721, 268815, 207898, 915494, 410667, 250412, 906810, 587837, 863823, 
           222292, 254551, 192599, 395358, 448099, 232047, 658544, 813683, 618107, 716412, 879741, 
           574079, 603264, 428672, 869507, 830096, 511635, 981139, 436884, 618660, 296614, 235691, 
           228013, 859311, 946869, 443584, 143041, 970436, 397509, 164057, 205540, 970475, 484598, 
           668408, 676095, 127746, 807690, 165132, 945935, 466708, 679189, 682776, 860968, 172854, 
           374583, 463163, 431426, 429894, 437244, 436048, 595794, 726358, 288091, 150879, 976738, 
           271205, 778600, 305517, 393584, 649587, 631669, 844152, 521593, 352122, 587652, 850823, 
           800138, 679819, 972174, 166798, 466837, 424854, 429469, 271787, 394672, 578483, 115656, 
           829900, 783312, 373719, 973784, 980446, 642014, 136678, 151015, 762351, 406008, 858108, 998398]

"""
def generate_unique_seeds():
    seeds = set()
    while len(seeds) < 100:
        seed = random.randint(100000, 999999)
        seeds.add(seed)
    lst_seeds.append(seed)
    return list(seeds)

lst_seeds = generate_unique_seeds()
print(lst_seeds)

seed_list = generate_unique_seeds()
"""

n = len(seed_list)

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
print(f"\nNote: All results below are average values and standard deviations across the simulations.")
print()

def compute_stats(result_list):
    keys = result_list[0].keys()
    stats = {}
    for key in keys:
        values = [result[key] for result in result_list]
        if isinstance(values[0], (int, float)):
            mean = sum(values) / n
            std = statistics.stdev(values)
            stats[key] = (mean, std)
        else:
            stats[key] = (values[0], None)  # Non-numeric, show only once
    return stats

average_generic = compute_stats(generic_results_list)
average_on = compute_stats(results_on_list)
average_off = compute_stats(results_off_list)

print("===== GENERIC RESULTS =====")
for key, (mean, std) in average_generic.items():
    if std is not None:
        print(f"{key}: {mean:.2f} ± {std:.2f}")
    else:
        print(f"{key}: {mean}")

print("\n===== NON-INNOVATION RESULTS =====")
for key, (mean, std) in average_off.items():
    if std is not None:
        print(f"{key}: {mean:.2f} ± {std:.2f}")
    else:
        print(f"{key}: {mean}")

print("\n===== INNOVATION RESULTS =====")
for key, (mean, std) in average_on.items():
    if std is not None:
        print(f"{key}: {mean:.2f} ± {std:.2f}")
    else:
        print(f"{key}: {mean}")


print("\n===== STATISTICAL COMPARISON (Innovation vs Non-Innovation) =====")
ordered_keys = [key for key in results_on_list[0] if key in results_off_list[0]]

for key in ordered_keys:
    try:
        values_on = [r[key] for r in results_on_list]
        values_off = [r[key] for r in results_off_list]
        
        # Ensure numerical comparison
        if isinstance(values_on[0], (int, float)) and isinstance(values_off[0], (int, float)):
            t_stat, p_value = ttest_rel(values_on, values_off)
            mean_diff = np.mean(np.array(values_on) - np.array(values_off))
            significance = " *" if p_value < 0.05 else ""
            print(f"{key}: mean difference = {mean_diff:.2f}, p-value = {p_value:.4f}{significance}")
    except KeyError:
        continue  # Skip any key that might be missing in either result
