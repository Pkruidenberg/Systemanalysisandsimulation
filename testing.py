import salabim as sim
import random
import numpy as np

# Global parameters
INNOVATION = True  # Toggle between old and new behavior
AGV_SPEED = 2  # meters per second (adjusted from PDL's 4 m/s based on report data)
MAX_CHARGE = 100  # Battery level maximum is 100%
CHARGE_THRESHOLD = 20  # Battery below 20% will require charging
CHARGE_RATE = 5  # Charge rate per time unit
SIM_DURATION = 1440  # Simulation duration in minutes (24 hours)

# Warehouse layout parameters
NUM_AISLES = 20
AISLE_LENGTH = 50  # meters
ENERGY_PER_METER = 0.042  # Wh per meter (from report data)


# Distributions
def order_interval():
    # Exponential distribution with mean 3.47 minutes (from report data)
    return np.random.exponential(3.47)


def items_per_order():
    # Poisson distribution with mean 5 (from report data)
    return max(1, np.random.poisson(5))


def pick_time_per_item():
    # Normal distribution with mean 1 minute and std 0.75 (from report data)
    return max(0.1, np.random.normal(1, 0.75))


def container_load_time():
    # Normal distribution for batch preparation time
    return max(0.5, np.random.normal(2, 1))


class OrderGenerator(sim.Component):
    def process(self):
        while True:
            # Generate orders with exponential inter-arrival times
            self.hold(order_interval())

            # Create a new order
            order = Order()
            order.num_items = items_per_order()
            order.setup()

            # Log order creation
            env.log_order_created(order)

            # Activate the order
            order.activate()


class Order(sim.Component):
    def setup(self):
        self.num_items = 0
        self.order_list = []
        self.creation_time = env.now()
        self.completion_time = None

        # Generate random items for this order
        for _ in range(self.num_items):
            # In a real implementation, we would select actual items
            # For simulation purposes, we'll just generate aisle numbers
            aisle = random.randint(1, NUM_AISLES)
            self.order_list.append(aisle)

        # Sort by aisle for efficient routing
        self.order_list.sort()

        # Remove duplicates (only visit each aisle once)
        self.order_list = list(dict.fromkeys(self.order_list))

    def process(self):
        # Notify ASRS that items need to be picked
        for aisle in self.order_list:
            for asrs in asrs_list:
                if asrs.aisle_number == aisle:
                    asrs.orders_to_process.append(self)
                    if INNOVATION:
                        asrs.activate()  # Start preparing container

        # Request an available AGV
        available_agvs = [agv for agv in agv_list if agv.is_idle]

        if available_agvs:
            my_agv = available_agvs[0]
            my_agv.is_idle = False
            my_agv.assigned_order = self
            my_agv.activate()
        else:
            # If no AGV is available, wait for one
            self.wait_for_agv = True
            self.passivate()

    def mark_completed(self):
        self.completion_time = env.now()
        env.log_order_completed(self)


class ASRS(sim.Component):
    def setup(self, aisle_number):
        self.aisle_number = aisle_number
        self.orders_to_process = []
        self.containers_ready = {}  # Order -> Container mapping

    def process(self):
        while True:
            # If innovation is on, prepare containers in advance
            if INNOVATION and self.orders_to_process:
                order = self.orders_to_process[0]

                # Create a container for this order if not already created
                if order not in self.containers_ready:
                    # Simulate time to prepare items for this aisle
                    items_in_aisle = sum(1 for aisle in order.order_list if aisle == self.aisle_number)
                    preparation_time = sum(pick_time_per_item() for _ in range(items_in_aisle))

                    self.hold(preparation_time)

                    # Create container and mark as ready
                    container = Container()
                    container.target_order = order
                    container.aisle = self.aisle_number
                    self.containers_ready[order] = container

            # Wait for next event
            self.passivate()


class AGV(sim.Component):
    def setup(self):
        self.battery = MAX_CHARGE
        self.assigned_order = None
        self.current_position = "depot"  # Start at depot
        self.is_idle = True
        self.total_distance = 0
        self.total_waiting_time = 0
        self.start_service_time = 0

    def process(self):
        while True:
            # Check if battery is low
            if self.battery < CHARGE_THRESHOLD:
                self.go_to_charger()

            # Wait until assigned an order
            if not self.assigned_order:
                self.is_idle = True
                self.passivate()

            self.is_idle = False
            self.start_service_time = env.now()

            # Choose pickup method based on innovation setting
            if not INNOVATION:
                self.sequential_pickup()
            else:
                self.container_pickup()

            # Mark order as completed
            self.assigned_order.mark_completed()

            # Reset for next order
            self.assigned_order = None
            self.is_idle = True

            # Check for waiting orders
            waiting_orders = [o for o in order_list if hasattr(o, 'wait_for_agv') and o.wait_for_agv]
            if waiting_orders:
                order = waiting_orders[0]
                order.wait_for_agv = False
                self.assigned_order = order
                self.is_idle = False
                order.activate()

    def sequential_pickup(self):
        # Implementation of the sequential pickup process
        for aisle in self.assigned_order.order_list:
            # Drive to the ASRS at this aisle
            self.drive_to_asrs(aisle)

            # Find the ASRS for this aisle
            asrs = None
            for a in asrs_list:
                if a.aisle_number == aisle:
                    asrs = a
                    break

            if asrs:
                # Calculate items to pick from this aisle
                items_in_aisle = sum(1 for a in self.assigned_order.order_list if a == aisle)

                # Wait for items to be picked (simulate ASRS operation)
                wait_start = env.now()
                for _ in range(items_in_aisle):
                    self.hold(pick_time_per_item())
                self.total_waiting_time += env.now() - wait_start

        # Drive to drop-off point
        self.drive_to_drop_off()

    def container_pickup(self):
        # Implementation of the container-based batch pickup process
        for aisle in self.assigned_order.order_list:
            # Drive to the ASRS at this aisle
            self.drive_to_asrs(aisle)

            # Find the ASRS for this aisle
            asrs = None
            for a in asrs_list:
                if a.aisle_number == aisle:
                    asrs = a
                    break

            if asrs:
                # Check if container is already prepared
                if self.assigned_order in asrs.containers_ready:
                    container = asrs.containers_ready[self.assigned_order]
                    # Container is ready, just transfer it (quick operation)
                    self.hold(0.5)  # Quick transfer time
                    # Remove from ready containers
                    del asrs.containers_ready[self.assigned_order]
                else:
                    # Container not ready, need to wait for preparation
                    wait_start = env.now()

                    # Calculate items to pick from this aisle
                    items_in_aisle = sum(1 for a in self.assigned_order.order_list if a == aisle)

                    # Create container and wait for loading
                    container = Container()
                    container.target_order = self.assigned_order
                    container.aisle = aisle

                    # Wait for container to be loaded
                    self.hold(sum(pick_time_per_item() for _ in range(items_in_aisle)))

                    self.total_waiting_time += env.now() - wait_start

                # Remove order from ASRS queue if this was the last item
                if asrs in [a for a in asrs_list if self.assigned_order in a.orders_to_process]:
                    asrs.orders_to_process.remove(self.assigned_order)

        # Drive to drop-off point
        self.drive_to_drop_off()

    def drive_to_asrs(self, aisle):
        # Calculate distance to the ASRS based on warehouse layout
        # This is a simplified version; in reality, would use the distance matrix
        if self.current_position == "depot":
            distance = aisle * 10  # Example distance calculation
        elif self.current_position == "charger":
            distance = aisle * 10 + 5  # Example with offset for charger
        else:
            # Distance between aisles
            distance = abs(int(self.current_position) - aisle) * 10

        # Update total distance
        self.total_distance += distance

        # Calculate travel time based on distance and AGV speed
        travel_time = distance / AGV_SPEED

        # Update battery consumption (0.042 Wh per meter from report)
        self.battery -= distance * ENERGY_PER_METER

        # Hold for travel time
        self.hold(travel_time)

        # Update current position
        self.current_position = aisle

    def drive_to_drop_off(self):
        # Calculate distance to drop-off point
        if self.current_position == "depot" or self.current_position == "charger":
            distance = 0  # Already at depot
        else:
            # Distance from current aisle to depot
            distance = int(self.current_position) * 10

        # Update total distance
        self.total_distance += distance

        # Calculate travel time
        travel_time = distance / AGV_SPEED

        # Update battery consumption
        self.battery -= distance * ENERGY_PER_METER

        # Hold for travel time
        self.hold(travel_time)

        # Update current position
        self.current_position = "depot"

    def go_to_charger(self):
        # Calculate distance to charger
        if self.current_position == "charger":
            distance = 0  # Already at charger
        elif self.current_position == "depot":
            distance = 5  # Example distance from depot to charger
        else:
            # Distance from current aisle to charger
            distance = int(self.current_position) * 10 + 5  # Example with offset

        # Update total distance
        self.total_distance += distance

        # Calculate travel time
        travel_time = distance / AGV_SPEED

        # Update battery consumption
        self.battery -= distance * ENERGY_PER_METER

        # Hold for travel time
        self.hold(travel_time)

        # Update current position
        self.current_position = "charger"

        # Find an available charger
        chargers = charger_list
        if chargers:
            charger = chargers[0]  # Take the first charger

            # Calculate charge time
            charge_time = (MAX_CHARGE - self.battery) / CHARGE_RATE

            # Hold for charging time
            self.hold(charge_time)

            # Battery is now full
            self.battery = MAX_CHARGE


class Container(sim.Component):
    def setup(self):
        self.target_order = None
        self.aisle = None


class Charger(sim.Component):
    def setup(self):
        self.my_queue = sim.Queue(f"Charger_{self.name()}_queue")

    def process(self):
        # Add a process method to make this an active component
        while True:
            # Wait for AGVs to charge
            self.passivate()


class WarehouseStats:
    def __init__(self):
        self.orders_created = 0
        self.orders_completed = 0
        self.total_order_time = 0
        self.total_agv_distance = 0
        self.total_agv_waiting_time = 0
        self.order_times = []

    def log_order_created(self, order):
        self.orders_created += 1

    def log_order_completed(self, order):
        self.orders_completed += 1
        order_time = order.completion_time - order.creation_time
        self.total_order_time += order_time
        self.order_times.append(order_time)

    def update_agv_stats(self, agvs):
        self.total_agv_distance = sum(agv.total_distance for agv in agvs)
        self.total_agv_waiting_time = sum(agv.total_waiting_time for agv in agvs)

    def print_stats(self):
        print("\n===== Warehouse Simulation Statistics =====")
        print(f"Innovation mode: {'ON' if INNOVATION else 'OFF'}")
        print(f"Simulation time: {env.now():.2f} minutes")
        print(f"Orders created: {self.orders_created}")
        print(f"Orders completed: {self.orders_completed}")

        if self.orders_completed > 0:
            avg_order_time = self.total_order_time / self.orders_completed
            print(f"Average order completion time: {avg_order_time:.2f} minutes")

            throughput = self.orders_completed / (env.now() / 60)  # Orders per hour
            print(f"Throughput: {throughput:.2f} orders/hour")

        print(f"Total AGV distance traveled: {self.total_agv_distance:.2f} meters")
        print(f"Total AGV waiting time: {self.total_agv_waiting_time:.2f} minutes")

        if self.order_times:
            print(f"Minimum order time: {min(self.order_times):.2f} minutes")
            print(f"Maximum order time: {max(self.order_times):.2f} minutes")


# Create simulation environment
env = sim.Environment(trace=False)
env.stats = WarehouseStats()
env.log_order_created = env.stats.log_order_created
env.log_order_completed = env.stats.log_order_completed

# Initialize lists to keep track of components (instead of using components_by_class)
asrs_list = []
agv_list = []
charger_list = []
order_list = []

# Initialize components
# Create ASRS systems (one per aisle)
for i in range(1, NUM_AISLES + 1):
    asrs = ASRS(name=f"ASRS_{i}", aisle_number=i)
    asrs.activate()
    asrs_list.append(asrs)

# Create AGVs
agvs = []
for i in range(10):  # Assume 10 AGVs
    agv = AGV()
    agv.battery = MAX_CHARGE
    agv.activate()
    agv_list.append(agv)
    agvs.append(agv)  # Keep a separate list for statistics

# Create chargers
for i in range(3):  # Assume 3 charging stations
    charger = Charger()
    charger.activate()
    charger_list.append(charger)

# Create order generator
order_generator = OrderGenerator()
order_generator.activate()

# Run the simulation
print(f"Starting simulation with innovation mode {'ON' if INNOVATION else 'OFF'}")
env.run(till=SIM_DURATION)

# Update and print statistics
env.stats.update_agv_stats(agvs)
env.stats.print_stats()
