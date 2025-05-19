# -*- coding: utf-8 -*-
"""
Created on Mon May 19 14:34:24 2025

@author: Patrick Kruidenberg
"""
import salabim as sim
import numpy as np
import csv

# Enable process mode
sim.yieldless(False)

# Parameters
CSV_FILE = 'warehouse_items.csv'
exp_lambda = 0.288
IAT = 1 / exp_lambda * 60  # seconds
mean_order_size = 5

# Load item data
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

# Order Generator
class TOrderGenerator(sim.Component):
    def setup(self, IAT, mean_order_size, item_pool):
        self.InterArrivalTime = sim.Exponential(IAT)
        self.ProductsPerOrder = sim.Poisson(mean_order_size)
        self.item_pool = item_pool

    def process(self):
        while True:
            yield self.hold(self.InterArrivalTime.sample())
            NrP = int(self.ProductsPerOrder.sample())
            selected_items = np.random.choice(self.item_pool, size=NrP, replace=False).tolist()
            timestamp = self.env.now()
            NewOrder = TOrder(NrProd=NrP, items=selected_items, timestamp=timestamp)
            NewOrder.enter(AllOrders)
            dispatcher.activate()  # Wake dispatcher

# Order class
class TOrder(sim.Component):
    order_counter = 0

    def setup(self, NrProd, items, timestamp):
        TOrder.order_counter += 1
        self.OrderID = TOrder.order_counter
        self.NrProd = NrProd
        self.Items = items
        self.timestamp = timestamp

        def extract_aisle_num(aisle_str):
            return int(aisle_str.split('-')[1])

        self.LoadPlan = sorted(set((item['AisleLocation']) for item in self.Items))
        self.ItemsByAisle = {}
        for item in self.Items:
            aisle_num = extract_aisle_num(item['Aisle'])
            if aisle_num not in self.ItemsByAisle:
                self.ItemsByAisle[aisle_num] = []
            self.ItemsByAisle[aisle_num].append(item)

        print(f"[Order {self.OrderID}] Created at {timestamp:.1f}s with {NrProd} items across aisles {self.LoadPlan}")

# Dispatcher
class TDispatcher(sim.Component):
    def setup(self, all_orders, available_agvs):
        self.AllOrders = all_orders
        self.AvailableAGVs = available_agvs

    def process(self):
        while True:
            if len(self.AllOrders) > 0 and len(self.AvailableAGVs) > 0:
                order = self.AllOrders.pop(0)
                agv = self.AvailableAGVs.pop(0)
                print(f"Dispatching Order {order.OrderID} to AGV {agv.ID}")
                agv.assign_order(order)
            else:
                yield self.passivate()

# ASRS loader
class TASRS(sim.Component):
    def setup(self, aisle, available_agvs):
        self.aisle = aisle
        self.queue = []
        self.available_agvs = available_agvs
        self.loading_time_mean = 60
        self.loading_time_std = 45

    def process(self):
        while True:
            if self.queue:
                agv = self.queue.pop(0)
                order = agv.current_order
                items_in_aisle = len(order.ItemsByAisle.get(self.aisle, []))
                if items_in_aisle > 0:
                    load_time_per_item = max(0, np.random.normal(self.loading_time_mean, self.loading_time_std))
                    total_load_time = items_in_aisle * load_time_per_item
                    print(f"ASRS {self.aisle}: Loading AGV {agv.ID} for {items_in_aisle} items, will take {total_load_time:.1f}s")
                    yield self.hold(total_load_time)
                else:
                    print(f"ASRS {self.aisle}: No items to load for AGV {agv.ID}")
                agv.asrs_loaded(self.aisle)
            else:
                yield self.passivate()

    def enqueue_agv(self, agv):
        self.queue.append(agv)
        self.activate()

# AGV class
class TAGV(sim.Component):
    def setup(self, ID, dispatcher, available_agvs, asrs_dict):
        self.ID = ID
        self.dispatcher = dispatcher
        self.available_agvs = available_agvs
        self.asrs_dict = asrs_dict
        self.current_order = None
        self.aisles_to_load = []
        self.aisles_loaded = set()

    def assign_order(self, order):
        self.current_order = order
        self.activate()

    def asrs_loaded(self, aisle):
        self.aisles_loaded.add(aisle)
        if self.aisles_loaded == set(self.aisles_to_load):
            self.activate()

    def process(self):
        while True:
            if self.current_order is None:
                if self not in self.available_agvs:
                    self.available_agvs.append(self)
                    self.dispatcher.activate()
                yield self.passivate()
            else:
                print(f"AGV {self.ID} starting Order {self.current_order.OrderID}")
                self.aisles_to_load = list(self.current_order.ItemsByAisle.keys())
                self.aisles_loaded = set()
                for aisle in self.aisles_to_load:
                    self.asrs_dict[aisle].enqueue_agv(self)
                while self.aisles_loaded != set(self.aisles_to_load):
                    yield self.passivate()
                print(f"AGV {self.ID} completed loading for Order {self.current_order.OrderID}")
                travel_time = len(self.current_order.LoadPlan) * 10
                yield self.hold(travel_time)
                print(f"AGV {self.ID} completed delivery for Order {self.current_order.OrderID}")
                self.current_order = None
                self.activate()

# Initialize components
AllOrders = sim.Queue(name='AllOrders')
env = sim.Environment(trace=False)
available_agvs = []
aisle_numbers = set(int(item['Aisle'].split('-')[1]) for item in items)
asrs_dict = {}

for aisle_num in aisle_numbers:
    asrs = TASRS(name=f"ASRS_{aisle_num}", aisle=aisle_num, available_agvs=available_agvs)
    asrs_dict[aisle_num] = asrs
    asrs.activate()

agvs = []
for i in range(3):
    agv = TAGV(name=f"AGV_{i+1}", ID=i+1, dispatcher=None, available_agvs=available_agvs, asrs_dict=asrs_dict)
    agv.activate()
    agvs.append(agv)

dispatcher = TDispatcher(name="Dispatcher", all_orders=AllOrders, available_agvs=available_agvs)
dispatcher.activate()

for agv in agvs:
    agv.dispatcher = dispatcher

order_generator = TOrderGenerator(name="OrderGenerator", IAT=IAT, mean_order_size=mean_order_size, item_pool=items)
order_generator.activate()

env.run(till=3600)
