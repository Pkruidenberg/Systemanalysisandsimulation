# -*- coding: utf-8 -*-
"""
Created on Mon May 19 14:34:24 2025

@author: Patrick Kruidenberg
"""
import salabim as sim
import numpy as np
import csv

# Enable process mode (important!)
sim.yieldless(False)

# Parameters (add as needed)
CSV_FILE = 'warehouse_items.csv'
exp_lambda = 0.288  # Exponential value in minutes
IAT = 1 / exp_lambda * 60  # Inter arrival time in seconds
mean_order_size = 5  # Poisson mean

# Loading items from the csv
items = []

with open(CSV_FILE, mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        items.append({
            'ItemID': row['ItemID'],
            'ItemName': row['ItemName'],
            'Category': row['Category'],
            'Aisle': row['Aisle'],
            'AisleLocation': int(row['AisleLocation'])
        })

print(f"Loaded {len(items)} items from {CSV_FILE}")


# Class for generating orders over time
class TOrderGenerator(sim.Component):
    def setup(self, IAT, mean_order_size, item_pool):
        # Set up arrival and order size distributions
        self.InterArrivalTime = sim.Exponential(IAT)         # Time between orders
        self.ProductsPerOrder = sim.Poisson(mean_order_size) # Number of products per order
        self.item_pool = item_pool                           # Full list of possible items

    def process(self):
        while True:
            myIAT = self.InterArrivalTime.sample()  # Sample time until next order
            self.hold(myIAT)                         # Wait that amount of simulated time

            NrP = int(self.ProductsPerOrder.sample())  # Sample how many items in this order
            
            # Randomly choose unique items for this order
            selected_items = np.random.choice(
                self.item_pool, size=NrP, replace=False
            ).tolist()
            
            # Capture the current simulation time as the timestamp
            timestamp = self.env.now()

            # Create a new order with sampled items and timestamp
            NewOrder = TOrder(NrProd=NrP, items=selected_items, timestamp=timestamp)
            NewOrder.enter(AllOrders)  # Place the order in the global queue

            yield  # Yield to allow simulation time to proceed

# Class representing a single order
class TOrder(sim.Component):
    def setup(self, NrProd, items, timestamp):
        self.NrProd = NrProd
        self.Items = items  # List of item dicts
        self.timestamp = timestamp  # Time order was created



# Create a queue to hold generated orders
AllOrders = sim.Queue(name='AllOrders')

# Create the environment
env = sim.Environment(trace=False)  # Set trace=True for step-by-step logging

# Start the order generator
order_generator = TOrderGenerator(IAT=IAT, mean_order_size=mean_order_size, item_pool=items)

# Run the simulation for 1 hour (3600 seconds) as a test
env.run(till=3600)

# After simulation, print some results
print(f"\nSimulation complete.")
print(f"Total orders generated: {AllOrders.length()}")
print(f"First few orders (with item info):")

for i, order in enumerate(AllOrders):
    print(f"\nOrder {i+1}: {order.NrProd} items, created at {order.timestamp:.2f} seconds")
    for item in order.Items:
        print(f" - {item['ItemID']} ({item['Aisle']}, Location {item['AisleLocation']})")
    if i >= 4:
        break  # Only show first 5 orders




    
    