import random
import csv
random.seed()

def generate_unique_seeds():
    seeds = set()
    while len(seeds) < 100:
        seed = random.randint(100000, 999999)
        seeds.add(seed)

    seed_list = list(seeds)

    # Write to CSV file
    with open('seeds.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['seed'])  # Header
        for seed in seed_list:
            writer.writerow([seed])

    return seed_list
generate_unique_seeds()
