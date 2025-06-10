import salabim as sim
import random
import csv
import numpy as np
import pandas as pd

# ===============================================================================
# GLOBAL PARAMETERS - Main settings that control the entire simulation
# ===============================================================================

INNOVATION = True                  # Switch between old method (False) and new container method (True)
AGV_SPEED = 2                       # How fast AGVs move in meters per second
MAX_CHARGE = 100                    # Full battery is 100%
CHARGE_THRESHOLD = 45               # AGVs need charging when battery drops below 45%
CHARGE_RATE = 4.5 / 60              # How fast batteries charge (4.5% per minute converted to per second)
SIM_DURATION = 24 * 60 * 60         # How long simulation runs (24 hours in seconds)
CSV_FILE = 'warehouse_items.csv'    # File containing all warehouse items
ADJACENCY_CSV = 'adjacency.csv'     # File showing which nodes connect to which nodes
AGV_NUMBR = 3                       # Set the number of AGVs
arrival_time_seconds = 17.3        # 17.3 seconds between orders,  means 5000 orders for the day
#arrival_time_seconds = 208.2        # 208.2 seconds between orders, means 415 orders per day
SEED = 43
random.seed(SEED)                   # For salabim distributions and Python random
np.random.seed(SEED)                # For numpy-based selections



# ===============================================================================
# ANIMATION SETTINGS - Controls how the warehouse looks on screen
# ===============================================================================

class AnimationConfig:
    # Window size
    WINDOW_WIDTH = 1600
    WINDOW_HEIGHT = 950

    # Layout measurements
    AISLE_WIDTH = 400  # How wide each aisle appears
    AISLE_HEIGHT = 30  # How tall each aisle appears
    NODE_RADIUS = 12  # Size of dots representing locations
    AGV_SIZE = 15  # Size of AGV squares
    FONT_SIZE = 10  # Text size
    AGV_TRAVEL_OFFSET = 25  # How far below aisles AGVs travel

    # Where things are positioned
    LEFT_COLUMN_X = 250  # X position of left aisles
    RIGHT_COLUMN_X = 800  # X position of right aisles
    AISLE_SPACING_Y = 55  # Vertical space between aisles
    TOP_MARGIN = 80  # Space from top of screen

    # Colors for different things
    AISLE_COLOR = "lightgray"
    NODE_COLOR = "lightblue"
    DEPOT_COLOR = "orange"  # Where AGVs start
    DROPOFF_COLOR = "orange"  # Where AGVs deliver orders
    CHARGER_COLOR = "darkgreen"  # Charging station
    ASRS_ACTIVE_COLOR = "firebrick"  # ASRS when working
    ASRS_IDLE_COLOR = "orange"  # ASRS when not working
    TEXT_COLOR = "black"

    # AGV colors based on what they're doing
    AGV_IDLE = "green"  # AGV waiting for work
    AGV_MOVING = "orange"  # AGV traveling
    AGV_PICKING = "red"  # AGV getting items
    AGV_QUEUED = "orange"  # AGV waiting in line
    AGV_CHARGING = "blue"  # AGV charging battery

    # Container colors
    CONTAINER_READY = "green"  # Container has items ready
    CONTAINER_EMPTY = "orange"  # Container is empty

# ===============================================================================
# WAREHOUSE LAYOUT SETTINGS - Physical warehouse structure
# ===============================================================================

NUM_AISLES = 20  # Total number of aisles in warehouse
AISLE_LENGTH = 50  # How long each aisle is in meters
ENERGY_PER_METER = 0.042  # How much battery AGVs use per meter traveled

# Special locations in warehouse
DEPOT_NODE = 0  # Where AGVs start and return to
DROP_OFF_NODE = 61  # Where AGVs deliver completed orders
CHARGER_NODE = 62  # Where AGVs go to charge batteries

# ===============================================================================
# WAREHOUSE SYSTEM - Main brain that coordinates everything
# ===============================================================================

class WarehouseSystem:
    """
    This is the main coordinator that keeps track of everything happening.
    Think of it as the warehouse manager that knows:
    - Which orders are being prepared where
    - Which ASRS machines are available
    - Which AGVs are working
    - Who the dispatcher is
    """
    def __init__(self):
        self.order_states = {}  # Tracks what's happening with each order at each location
        self.asrs_dict = {}  # List of all ASRS machines and where they are
        self.agv_list = []  # List of all AGVs
        self.dispatcher = None  # The dispatcher that assigns work
        # Order states can be: 'not_started', 'preparing', 'ready', 'transferred'

    def register_asrs(self, node, asrs):
        """Add an ASRS machine to our list so we know it exists"""
        self.asrs_dict[node] = asrs

    def register_agv(self, agv):
        """Add an AGV to our list so we can track it"""
        self.agv_list.append(agv)

    def register_dispatcher(self, dispatcher):
        """Tell the system who the dispatcher is"""
        self.dispatcher = dispatcher

    def can_start_preparation(self, order_id, node):
        """Check if we can start preparing items for an order at a specific location"""
        if order_id not in self.order_states:
            self.order_states[order_id] = {}

        current_state = self.order_states[order_id].get(node, 'not_started')
        return current_state == 'not_started'

    def set_order_state(self, order_id, node, state):
        """Update what stage an order is at for a specific location"""
        if order_id not in self.order_states:
            self.order_states[order_id] = {}
        self.order_states[order_id][node] = state

        SimLogger.log(f"Order state changed to '{state}' at node {node}",
                      order_id=order_id, component_type="WarehouseSystem")

    def get_order_state(self, order_id, node):
        """Find out what stage an order is at for a specific location"""
        return self.order_states.get(order_id, {}).get(node, 'not_started')

    def cleanup_completed_order(self, order_id):
        """Remove tracking info for orders that are finished"""
        if order_id in self.order_states:
            del self.order_states[order_id]

# Create the main warehouse system that everyone will use
warehouse_system = WarehouseSystem()

# ===============================================================================
# COORDINATE SYSTEM - Converts node numbers to screen positions
# ===============================================================================

def get_node_coordinates(node_number):
    """
    Convert a node number to x,y coordinates on screen.
    Each aisle has 3 nodes: entry, middle (where ASRS is), and exit.
    """
    # Special locations
    if node_number == DEPOT_NODE:
        return (AnimationConfig.RIGHT_COLUMN_X - AnimationConfig.AISLE_WIDTH // 2, 30)
    elif node_number == DROP_OFF_NODE:
        return (AnimationConfig.LEFT_COLUMN_X + AnimationConfig.AISLE_WIDTH // 2, 30)
    elif node_number == CHARGER_NODE:
        return ((AnimationConfig.RIGHT_COLUMN_X - AnimationConfig.AISLE_WIDTH // 2 +
                 AnimationConfig.LEFT_COLUMN_X + AnimationConfig.AISLE_WIDTH // 2) // 2, 30)
    else:
        # Calculate which aisle and position within that aisle
        aisle_num = ((node_number - 1) // 3) + 1
        position_in_aisle = (node_number - 1) % 3  # 0=entry, 1=middle, 2=exit

        # Calculate aisle coordinates - U-shaped layout
        if aisle_num <= NUM_AISLES // 2:  # Right column (aisles 1-10)
            base_x = AnimationConfig.RIGHT_COLUMN_X
            base_y = AnimationConfig.TOP_MARGIN + (aisle_num - 1) * AnimationConfig.AISLE_SPACING_Y
        else:  # Left column (aisles 11-20)
            base_x = AnimationConfig.LEFT_COLUMN_X
            base_y = AnimationConfig.TOP_MARGIN + (NUM_AISLES - aisle_num) * AnimationConfig.AISLE_SPACING_Y

        # Snake pattern for both columns
        if aisle_num <= NUM_AISLES // 2:  # Right column: snake pattern
            if aisle_num % 2 == 1:  # Odd aisles (1, 3, 5, 7, 9): LEFT TO RIGHT
                if position_in_aisle == 0:  # Entry node (left side)
                    return (base_x - AnimationConfig.AISLE_WIDTH // 2, base_y)
                elif position_in_aisle == 1:  # Middle node (center - ASRS location)
                    return (base_x, base_y)
                else:  # Exit node (right side)
                    return (base_x + AnimationConfig.AISLE_WIDTH // 2, base_y)
            else:  # Even aisles (2, 4, 6, 8, 10): RIGHT TO LEFT
                if position_in_aisle == 0:  # Entry node (right side)
                    return (base_x + AnimationConfig.AISLE_WIDTH // 2, base_y)
                elif position_in_aisle == 1:  # Middle node (center)
                    return (base_x, base_y)
                else:  # Exit node (left side)
                    return (base_x - AnimationConfig.AISLE_WIDTH // 2, base_y)
        else:  # Left column: snake pattern
            if aisle_num % 2 == 1:  # Odd aisles (11, 13, 15, 17, 19): RIGHT TO LEFT
                if position_in_aisle == 0:  # Entry node (right side)
                    return (base_x + AnimationConfig.AISLE_WIDTH // 2, base_y)
                elif position_in_aisle == 1:  # Middle node (center)
                    return (base_x, base_y)
                else:  # Exit node (left side)
                    return (base_x - AnimationConfig.AISLE_WIDTH // 2, base_y)
            else:  # Even aisles (12, 14, 16, 18, 20): LEFT TO RIGHT
                if position_in_aisle == 0:  # Entry node (left side)
                    return (base_x - AnimationConfig.AISLE_WIDTH // 2, base_y)
                elif position_in_aisle == 1:  # Middle node (center)
                    return (base_x, base_y)
                else:  # Exit node (right side)
                    return (base_x + AnimationConfig.AISLE_WIDTH // 2, base_y)

def get_asrs_color(node_number):
    """Return color based on whether ASRS is working or not"""
    asrs = warehouse_system.asrs_dict.get(node_number)
    if asrs and asrs.is_active:
        return AnimationConfig.ASRS_ACTIVE_COLOR
    else:
        return AnimationConfig.ASRS_IDLE_COLOR

# ===============================================================================
# ANIMATION HELPER FUNCTIONS - Create visual elements on screen
# ===============================================================================

def create_node_animation(node_number, node_type="normal"):
    """Create a visual dot and label for a location in the warehouse"""
    x, y = get_node_coordinates(node_number)

    # Choose colors and sizes based on what type of location this is
    if node_type == "asrs":
        color = lambda t: get_asrs_color(node_number)
        radius = AnimationConfig.NODE_RADIUS
        line_color = "maroon"
    elif node_type == "depot":
        color = AnimationConfig.DEPOT_COLOR
        radius = AnimationConfig.NODE_RADIUS + 3
        line_color = "maroon"
    elif node_type == "dropoff":
        color = AnimationConfig.DROPOFF_COLOR
        radius = AnimationConfig.NODE_RADIUS + 3
        line_color = "maroon"
    elif node_type == "charger":
        color = AnimationConfig.CHARGER_COLOR
        radius = AnimationConfig.NODE_RADIUS + 3
        line_color = "limegreen"
    else:
        color = AnimationConfig.NODE_COLOR
        radius = AnimationConfig.NODE_RADIUS
        line_color = "darkblue"

    # Create the circle
    sim.AnimateCircle(
        radius=radius,
        x=x, y=y,
        fillcolor=color,
        linecolor=line_color,
        linewidth=2
    )

    # Create the number label
    text_color = "white" if node_type == "charger" else AnimationConfig.TEXT_COLOR
    fontsize = 12

    sim.AnimateText(
        text=str(node_number if node_number != CHARGER_NODE else "62"),
        x=x, y=y,
        fontsize=fontsize,
        textcolor=text_color,
        text_anchor="c"
    )

def create_aisle_animation(aisle_num):
    """Create visual representation of one complete aisle with its 3 nodes"""
    # Calculate where this aisle should be positioned
    if aisle_num <= NUM_AISLES // 2:
        x = AnimationConfig.RIGHT_COLUMN_X
        y = AnimationConfig.TOP_MARGIN + (aisle_num - 1) * AnimationConfig.AISLE_SPACING_Y
    else:
        x = AnimationConfig.LEFT_COLUMN_X
        y = AnimationConfig.TOP_MARGIN + (NUM_AISLES - aisle_num) * AnimationConfig.AISLE_SPACING_Y

    # Draw the aisle background rectangle
    sim.AnimateRectangle(
        spec=(x - AnimationConfig.AISLE_WIDTH // 2, y - AnimationConfig.AISLE_HEIGHT // 2,
              x + AnimationConfig.AISLE_WIDTH // 2, y + AnimationConfig.AISLE_HEIGHT // 2),
        fillcolor=AnimationConfig.AISLE_COLOR,
        linecolor="black",
        linewidth=2
    )

    # Calculate node numbers for this aisle and create them
    entry_node = (aisle_num - 1) * 3 + 1
    middle_node = (aisle_num - 1) * 3 + 2  # This is where the ASRS is
    exit_node = (aisle_num - 1) * 3 + 3

    create_node_animation(entry_node, "normal")
    create_node_animation(middle_node, "asrs")  # ASRS location
    create_node_animation(exit_node, "normal")

def setup_animation():
    """Create all the visual elements for the warehouse"""
    # Create all aisles
    for aisle_num in range(1, NUM_AISLES + 1):
        create_aisle_animation(aisle_num)

    # Create special locations
    create_node_animation(DEPOT_NODE, "depot")
    create_node_animation(DROP_OFF_NODE, "dropoff")
    create_node_animation(CHARGER_NODE, "charger")

# ===============================================================================
# PATH FINDING SYSTEM - Loads warehouse layout and finds shortest routes
# ===============================================================================

def load_adjacency_matrix():
    """
    Load the map of how locations connect to each other.
    This tells us which nodes are connected and how far apart they are.
    """
    try:
        adj_df = pd.read_csv(ADJACENCY_CSV, index_col=0)
        adjacency_dict = {}

        for from_node in adj_df.index:
            from_node_int = int(from_node)
            adjacency_dict[from_node_int] = {}

            for to_node in adj_df.columns:
                to_node_int = int(to_node)
                if pd.notna(adj_df.loc[from_node, to_node]) and adj_df.loc[from_node, to_node] != 0:
                    distance_value = float(adj_df.loc[from_node, to_node])
                    adjacency_dict[from_node_int][to_node_int] = distance_value

        SimLogger.log(f"Loaded adjacency matrix with {len(adj_df)} nodes")
        return adjacency_dict
    except FileNotFoundError:
        SimLogger.log(f"Warning: {ADJACENCY_CSV} not found.")
        raise

def dijkstra_shortest_path(start, end, adjacency_dict):
    """Find the shortest path between two locations using Dijkstra's algorithm"""
    if start == end:
        return [start]

    # Initialize distances to all nodes as infinite
    distances = {node: float('inf') for node in adjacency_dict.keys()}
    distances[start] = 0
    previous = {}
    unvisited = set(adjacency_dict.keys())

    while unvisited:
        # Find unvisited node with smallest distance
        current = min(unvisited, key=lambda node: distances[node])

        if distances[current] == float('inf'):
            break

        if current == end:
            break

        unvisited.remove(current)

        # Check all neighbors of current node
        for neighbor, weight in adjacency_dict.get(current, {}).items():
            if neighbor in unvisited:
                new_distance = distances[current] + weight

                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous[neighbor] = current

    # Build the path by working backwards
    if end not in previous and end != start:
        return []

    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous.get(current)

    path.reverse()
    return path

def calculate_distance_between_nodes(from_node, to_node, adjacency_dict):
    """Calculate distance between two connected locations"""
    try:
        special_nodes = {DEPOT_NODE, DROP_OFF_NODE, CHARGER_NODE}

        # Special locations are close to each other
        if from_node in special_nodes and to_node in special_nodes:
            return 0

        # Charger is at same location as depot
        lookup_from = DEPOT_NODE if from_node == CHARGER_NODE else from_node
        lookup_to = DEPOT_NODE if to_node == CHARGER_NODE else to_node

        if lookup_from in adjacency_dict and lookup_to in adjacency_dict[lookup_from]:
            return adjacency_dict[lookup_from][lookup_to]
        else:
            # Fallback distance calculation
            return abs(lookup_from - lookup_to) * 10
    except:
        return abs(from_node - to_node) * 10

# ===============================================================================
# DATA LOADING SYSTEM - Loads warehouse items from files
# ===============================================================================

def load_warehouse_data():
    """Load list of all items in the warehouse from CSV file"""
    items = []
    try:
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
        SimLogger.log(f"Loaded {len(items)} items from {CSV_FILE}")
        return items
    except FileNotFoundError:
        SimLogger.log(f"Warning: {CSV_FILE} not found. Using simulated data.")
        return generate_simulated_items()

def generate_simulated_items():
    """Create fake items if the real item file is missing"""
    items = []
    for i in range(1000):
        aisle_num = random.randint(1, NUM_AISLES)
        aisle_location = (aisle_num - 1) * 3 + 2  # Middle node of aisle (where ASRS is)
        items.append({
            'ItemID': f'ITEM_{i:04d}',
            'ItemName': f'Product_{i}',
            'Category': random.choice(['Electronics', 'Clothing', 'Books', 'Home']),
            'Aisle': f'Aisle-{aisle_num}',
            'AisleLocation': aisle_location
        })
    return items

# ===============================================================================
# ORDER GENERATOR - Creates new customer orders randomly (Precomputed for consistency)
# ===============================================================================

def generate_orders_ahead(seed, duration_seconds, items_pool):
    """Precompute all order arrival times and item selections"""
    random.seed(seed)
    np.random.seed(seed)

    inter_arrival_time_dist = sim.Exponential(arrival_time_seconds)
    items_per_order_dist = sim.Poisson(5)

    time = 0
    order_defs = []

    while time < duration_seconds:
        iat = inter_arrival_time_dist.sample()
        num_items = max(1, int(items_per_order_dist.sample()))
        item_indices = np.random.choice(len(items_pool), size=num_items, replace=False)
        selected_items = [items_pool[i] for i in item_indices]

        order_defs.append((time, selected_items))
        time += iat

    return order_defs


class OrderFeeder(sim.Component):
    """Feeds precomputed orders into the simulation at scheduled times"""
    def setup(self, order_definitions):
        self.order_definitions = order_definitions

    def process(self):
        for timestamp, items in self.order_definitions:
            wait_time = timestamp - env.now()
            if wait_time > 0:
                self.hold(wait_time)

            order = Order(items=items, timestamp=env.now())
            all_orders.append(order)
            env.stats.orders_created += 1

            if INNOVATION:
                order.trigger_container_preparation()

            order.enter(order_queue)
            dispatcher.activate()


# ===============================================================================
# ORDER CLASS - Represents a customer order
# ===============================================================================

class Order(sim.Component):
    """
    Represents one customer order with multiple items.
    Calculates the best path to collect all items.
    """
    order_counter = 0

    def setup(self, items, timestamp):
        Order.order_counter += 1
        self.order_id = Order.order_counter
        self.items = items  # List of items customer wants
        self.timestamp = timestamp  # When order was created
        self.completion_time = None
        self.is_completed = False

        # Group items by which aisle they're in
        self.items_by_node = {}
        nodes_to_visit = set()

        for item in self.items:
            pickup_node = item['AisleLocation']
            nodes_to_visit.add(pickup_node)
            if pickup_node not in self.items_by_node:
                self.items_by_node[pickup_node] = []
            self.items_by_node[pickup_node].append(item)

        # Calculate best path to visit all locations
        self.load_plan = self.calculate_complete_path_with_adjacency(list(nodes_to_visit))
        self.pickup_nodes = list(nodes_to_visit)
        self.num_items = len(items)

    def calculate_complete_path_with_adjacency(self, pickup_nodes):
        """Calculate the best route to visit all pickup locations"""
        if not pickup_nodes:
            return [DEPOT_NODE, DROP_OFF_NODE]

        # Visit nodes in order
        sorted_pickup_nodes = sorted(pickup_nodes)
        complete_path = [DEPOT_NODE]
        current_position = DEPOT_NODE

        # Add path to each pickup location
        for pickup_node in sorted_pickup_nodes:
            path_to_pickup = dijkstra_shortest_path(current_position, pickup_node, adjacency_matrix)

            if len(path_to_pickup) > 1:
                complete_path.extend(path_to_pickup[1:])
            elif len(complete_path) == 0:
                complete_path.extend(path_to_pickup)

            current_position = pickup_node

        # Add path to dropoff
        path_to_dropoff = dijkstra_shortest_path(current_position, DROP_OFF_NODE, adjacency_matrix)
        if len(path_to_dropoff) > 1:
            complete_path.extend(path_to_dropoff[1:])

        return complete_path

    def trigger_container_preparation(self):
        """
        NEW METHOD ONLY: Tell ASRS machines to start preparing items immediately.
        This happens before an AGV is even assigned.
        """
        node_details = []
        for node in self.pickup_nodes:
            items_at_node = self.items_by_node[node]
            item_names = [item['ItemName'] for item in items_at_node]
            node_details.append(f"{node}: {', '.join(item_names)}")

        SimLogger.log(f"Triggering item preparation for {len(self.pickup_nodes)} nodes: [{'], ['.join(node_details)}]",
                      order_id=self.order_id)

        # Tell each ASRS to start preparing items
        for node in self.pickup_nodes:
            if not warehouse_system.can_start_preparation(self.order_id, node):
                SimLogger.log(f"Preparation already in progress for node {node}, skipping",
                              order_id=self.order_id, component_type="Order")
                continue

            warehouse_system.set_order_state(self.order_id, node, 'preparing')

            # Find the ASRS at this location and give it work
            asrs = next((a for a in asrs_list if a.node_number == node), None)
            if asrs:
                request_data = {
                    'order_id': self.order_id,
                    'node_number': node,
                    'items_list': self.items_by_node[node],
                    'request_time': env.now(),
                    'num_items': len(self.items_by_node[node])
                }
                asrs.prep_process.preparation_queue.append(request_data)
                if asrs.prep_process.status() == sim.passive:
                    asrs.prep_process.activate()

    def process(self):
        """Orders don't do anything on their own - they wait to be processed"""
        self.passivate()

    def mark_completed(self):
        """Mark this order as finished and update statistics"""
        if not self.is_completed:
            self.completion_time = env.now()
            self.is_completed = True
            env.stats.orders_completed += 1
            order_time = self.completion_time - self.timestamp
            env.stats.total_order_time += order_time
            env.stats.order_times.append(order_time)
            warehouse_system.cleanup_completed_order(self.order_id)

# ===============================================================================
# DISPATCHER - Assigns orders to available AGVs
# ===============================================================================

class Dispatcher(sim.Component):
    """
    The dispatcher is like a manager that assigns work to AGVs.
    It looks at the queue of orders and the list of available AGVs,
    and matches them up when possible.
    """
    def setup(self):
        self.order_queue = order_queue  # Orders waiting to be assigned
        self.available_agvs = available_agvs  # AGVs ready for work

    def process(self):
        """Main dispatching loop - matches orders with capable AGVs"""
        while True:
            # Check if we have both orders and AGVs available
            if len(self.order_queue) > 0 and len(self.available_agvs) > 0:
                order = self.order_queue.head()  # Peek at next order

                # Find ANY available AGV (no battery check here)
                suitable_agv = self.available_agvs[0]  # Just take the first available

                # Assign order to AGV
                order = self.order_queue.pop()
                self.available_agvs.remove(suitable_agv)
                SimLogger.log(f"Dispatching to AGV {suitable_agv.agv_id}",
                              order_id=order.order_id, component_type="Dispatcher")
                suitable_agv.assign_order(order)
            else:
                self.passivate()  # Wait for orders or AGVs


# ===============================================================================
# ASRS CLASS - Automated Storage and Retrieval System
# ===============================================================================

class ASRS(sim.Component):
    """
    ASRS is the robot that picks items from shelves.
    Each aisle has one ASRS in the middle.
    The ASRS handles AGV requests and works with containers (in new method).
    """
    def setup(self, node_number):
        self.node_number = node_number  # Which aisle this ASRS is in
        self.agv_queue = sim.Queue(f"ASRS_Node_{node_number}_Queue")  # AGVs waiting here
        self.pick_time = sim.Normal(60, 45)  # How long to pick each item
        self.is_active = False  # Whether currently working
        self.current_picking_order = None  # Which order being picked
        self.current_picking_items = 0  # How many items being picked

        # Create container and preparation process (for new method)
        self.container = Container(node_number=self.node_number, asrs=self)
        self.prep_process = ASRSContainerInterface(node_number=self.node_number, asrs=self)
        self.prep_process.activate()

        # Setup visual display
        self.animation_manager = ASRSAnimationManager(self)

    def hand_items_to_container(self, order_id, items_list, prep_time=None):
        """Give picked items to the container (new method only)"""
        self.container.receive_items_from_asrs(order_id, items_list, prep_time)

    def process(self):
        """
        Main ASRS process - handles AGVs that arrive for pickup.
        Behavior depends on whether we're using old or new method.
        """
        while True:
            if self.agv_queue.length() > 0:
                agv = self.agv_queue.pop()

                agv.agv_status = "picking_up"

                # Move AGV to ASRS position for pickup
                asrs_x, asrs_y = get_node_coordinates(self.node_number)
                agv.animation_object.x = asrs_x
                agv.animation_object.y = asrs_y

                SimLogger.log(f"AGV {agv.agv_id} moved to ASRS position for pickup",
                              component_type="ASRS Node", component_id=self.node_number)

                order = agv.current_order

                if order and self.node_number in order.items_by_node:
                    if INNOVATION:
                        # NEW METHOD: Use containers for batch transfer
                        current_state = warehouse_system.get_order_state(order.order_id, self.node_number)

                        if current_state == 'ready':
                            # Container is ready, transfer immediately
                            self.container.request_transfer_to_agv(agv, order.order_id)
                            self.passivate()
                            env.stats.container_ready_transfers += 1
                        elif current_state == 'preparing':
                            # Container being prepared, wait for it
                            SimLogger.log(f"AGV {agv.agv_id} waiting for ongoing preparation",
                                          order_id=order.order_id, component_type="ASRS Node",
                                          component_id=self.node_number)
                            while warehouse_system.get_order_state(order.order_id, self.node_number) != 'ready':
                                self.hold(1)
                            self.container.request_transfer_to_agv(agv, order.order_id)
                            self.passivate()
                            env.stats.container_wait_transfers += 1
                        else:
                            # No preparation started, do it now
                            if warehouse_system.can_start_preparation(order.order_id, self.node_number):
                                self.is_active = True
                                items_at_node = len(order.items_by_node[self.node_number])
                                self.current_picking_order = order.order_id
                                self.current_picking_items = items_at_node

                                warehouse_system.set_order_state(order.order_id, self.node_number, 'preparing')

                                SimLogger.log(f"Container not ready, AGV {agv.agv_id} waiting for new preparation",
                                              order_id=order.order_id, component_type="ASRS Node",
                                              component_id=self.node_number)

                                # Pick items one by one
                                for i, item in enumerate(order.items_by_node[self.node_number]):
                                    item_prep_time = max(0, self.pick_time.sample())

                                    SimLogger.log(
                                        f"picking item {i + 1}/{items_at_node}: {item['ItemName']} (took {item_prep_time:.1f}s)",
                                        order_id=order.order_id, component_type="ASRS Node",
                                        component_id=self.node_number)

                                    self.hold(item_prep_time)

                                # Give all items to container
                                self.hand_items_to_container(order.order_id, order.items_by_node[self.node_number])
                                warehouse_system.set_order_state(order.order_id, self.node_number, 'ready')
                                self.container.request_transfer_to_agv(agv, order.order_id)
                                self.passivate()
                                env.stats.container_wait_transfers += 1
                            else:
                                # Wait for ongoing preparation
                                while warehouse_system.get_order_state(order.order_id, self.node_number) != 'ready':
                                    self.hold(1)
                                self.container.request_transfer_to_agv(agv, order.order_id)
                                self.passivate()
                    else:
                        # OLD METHOD: Pick items and give directly to AGV
                        self.is_active = True
                        items_at_node = order.items_by_node[self.node_number]
                        self.current_picking_order = order.order_id
                        self.current_picking_items = len(items_at_node)

                        SimLogger.log(
                            f"AGV {agv.agv_id} waiting for sequential preparation of {len(items_at_node)} items",
                            order_id=order.order_id, component_type="ASRS Node", component_id=self.node_number)

                        total_prep_time = 0
                        # Pick items one by one and give directly to AGV
                        for i, item in enumerate(items_at_node):
                            item_prep_time = max(0, self.pick_time.sample())
                            total_prep_time += item_prep_time

                            SimLogger.log(
                                f"picking item {i + 1}/{len(items_at_node)}: {item['ItemName']})",
                                order_id=order.order_id, component_type="ASRS Node", component_id=self.node_number)

                            self.hold(item_prep_time)

                        SimLogger.log(
                            f"completed sequential preparation of {len(items_at_node)} items (total picking took {total_prep_time:.1f}s)",
                            order_id=order.order_id, component_type="ASRS Node", component_id=self.node_number)

                        # Clear status and give items directly to AGV
                        self.current_picking_order = None
                        self.current_picking_items = 0
                        self.is_active = False

                        env.stats.sequential_transfers += 1
                        agv.loading_complete()  # Direct handoff to AGV

                SimLogger.log(f"AGV {agv.agv_id} moved back to travel position",
                              component_type="ASRS Node", component_id=self.node_number)
            else:
                self.passivate()

    def enqueue_agv(self, agv):
        """Add an AGV to the queue waiting for service"""
        agv.enter(self.agv_queue)
        self.activate()

# ===============================================================================
# CONTAINER CLASS - Stores items temporarily (new method only)
# ===============================================================================

class Container(sim.Component):
    """
    Container sits next to each ASRS and holds items until AGV arrives.
    This is only used in the new method (when INNOVATION = True).
    Think of it as a temporary storage box.
    """
    def setup(self, node_number, asrs):
        self.node_number = node_number  # Which aisle this container is in
        self.asrs = asrs  # The ASRS that fills this container
        self.items_ready = {}  # Items ready for each order
        self.transfer_queue = []  # AGVs waiting for items

    def receive_items_from_asrs(self, order_id, items_list, prep_time=None):
        """Receive items from ASRS and store them"""
        if order_id not in self.items_ready:
            self.items_ready[order_id] = []

        self.items_ready[order_id].extend(items_list)

        item_names = [item['ItemName'] for item in items_list]
        items_detail = ', '.join(item_names)

        if prep_time is not None:
            SimLogger.log(
                f"received {len(items_list)} item(s) from ASRS - Items: [{items_detail}] (took {prep_time:.1f}s)",
                order_id=order_id, component_type="Container", component_id=self.node_number)
        else:
            SimLogger.log(f"received {len(items_list)} item(s) from ASRS - Items: [{items_detail}]",
                          order_id=order_id, component_type="Container", component_id=self.node_number)

    def request_transfer_to_agv(self, agv, order_id):
        """AGV requests items from container"""
        self.transfer_queue.append((agv, order_id))
        self.activate()

    def process(self):
        """Handle transfers to AGVs"""
        while True:
            if self.transfer_queue:
                agv, order_id = self.transfer_queue.pop(0)

                if order_id in self.items_ready:
                    items_list = self.items_ready[order_id]
                    item_names = [item['ItemName'] for item in items_list]
                    items_detail = ', '.join(item_names)

                    # Take time to transfer items
                    transfer_time = 15
                    self.hold(transfer_time)

                    SimLogger.log(f"dropped items to AGV {agv.agv_id} - Items: [{items_detail}]",
                                  order_id=order_id, component_type="Container", component_id=self.node_number)

                    # Remove items from container
                    del self.items_ready[order_id]
                    warehouse_system.set_order_state(order_id, self.node_number, 'transferred')

                    # Tell AGV loading is complete
                    agv.loading_complete()
                    self.asrs.activate()
                else:
                    # No items ready, just complete loading
                    agv.loading_complete()
                    self.asrs.activate()
            else:
                self.passivate()

    def has_items_for_order(self, order_id):
        """Check if container has items ready for specific order"""
        return order_id in self.items_ready

    def has_items(self):
        """Check if container has any items at all"""
        return len(self.items_ready) > 0

# ===============================================================================
# ASRS CONTAINER INTERFACE - Handles item preparation (new method only)
# ===============================================================================

class ASRSContainerInterface(sim.Component):
    """
    This is the part of the ASRS that prepares items in advance.
    It works separately from the main ASRS process.
    When orders come in, this starts picking items immediately
    and puts them in containers, even before AGVs arrive.
    This is only used in the new method.
    """
    def setup(self, node_number, asrs):
        self.node_number = node_number  # Which aisle this is for
        self.asrs = asrs  # The ASRS machine this works with
        self.preparation_queue = []  # List of preparation requests

    def process(self):
        """Main process that prepares items in advance"""
        while True:
            # Wait for preparation requests
            while len(self.preparation_queue) == 0:
                self.passivate()

            prep_request = self.preparation_queue.pop(0)

            # Mark ASRS as busy
            self.asrs.is_active = True

            # Set status for display
            self.asrs.current_picking_order = prep_request['order_id']
            self.asrs.current_picking_items = prep_request['num_items']

            # Double-check we should still do this
            current_state = warehouse_system.get_order_state(prep_request['order_id'], self.node_number)
            if current_state != 'preparing':
                SimLogger.log(f"Skipping preparation - state changed to {current_state}",
                              order_id=prep_request['order_id'], component_type="ASRS Node",
                              component_id=self.node_number)
                # Clear status
                self.asrs.current_picking_order = None
                self.asrs.current_picking_items = 0
                self.asrs.is_active = False
                continue

            # Create detailed item list for logging
            item_names = [item['ItemName'] for item in prep_request['items_list']]
            items_detail = ', '.join(item_names)

            SimLogger.log(f"starting item preparation - Items: [{items_detail}]",
                          order_id=prep_request['order_id'], component_type="ASRS Node", component_id=self.node_number)

            # Pick items one by one and put in container
            items_list = prep_request['items_list']
            total_prep_time = 0

            for i, item in enumerate(items_list):
                item_prep_time = max(0, self.asrs.pick_time.sample())
                total_prep_time += item_prep_time

                SimLogger.log(f"picking item {i + 1}/{len(items_list)}: {item['ItemName']}",
                              order_id=prep_request['order_id'], component_type="ASRS Node",
                              component_id=self.node_number)

                self.hold(item_prep_time)  # Time to pick this item

                # Put item in container immediately
                self.asrs.hand_items_to_container(prep_request['order_id'], [item], item_prep_time)

            # Mark order as ready
            warehouse_system.set_order_state(prep_request['order_id'], self.node_number, 'ready')

            # Clear status
            self.asrs.current_picking_order = None
            self.asrs.current_picking_items = 0
            self.asrs.is_active = False

            SimLogger.log(f"completed preparation of {len(items_list)} items (total prep took {total_prep_time:.1f}s)",
                          order_id=prep_request['order_id'], component_type="ASRS Node", component_id=self.node_number)

# ===============================================================================
# AGV CLASS - Automated Guided Vehicle
# ===============================================================================

class AGV(sim.Component):
    """
    AGV is the robot vehicle that moves around the warehouse.
    It picks up items from ASRS machines and delivers them to the drop-off point.
    Each AGV has a battery that needs charging periodically.
    """

    def setup(self, agv_id):
        self.agv_id = agv_id
        self.battery = MAX_CHARGE
        self.current_order = None
        self.current_node = DEPOT_NODE
        self.total_distance = 0.0
        self.total_waiting_time = 0.0
        self.recharge_count = 0
        self.path_to_follow = []
        self.current_path_index = 0
        self.is_available = True
        self.agv_status = "idle"
        self.items_picked_up = 0
        self.total_items_in_order = 0

        # Setup animations using the manager
        self.animation_manager = AGVAnimationManager(self)

    def assign_order(self, order):
        self.current_order = order
        self.path_to_follow = order.load_plan.copy()
        self.current_path_index = 0
        self.is_available = False
        self.agv_status = "moving"
        self.items_picked_up = 0
        self.total_items_in_order = order.num_items
        self.activate()

    def process(self):
        """Main AGV operation loop with decentralized charging decisions"""
        while True:
            if self.current_order is None:
                # Make AGV available for new orders
                if not self.is_available:
                    self.is_available = True
                    self.agv_status = "idle"
                    available_agvs.append(self)
                    dispatcher.activate()
                self.passivate()
            else:
                # Execute order (existing code)
                if self.current_node != DEPOT_NODE:
                    self.return_to_depot()

                pickup_nodes = set(self.current_order.pickup_nodes)
                formatted_path = []
                for node in self.path_to_follow:
                    if node in pickup_nodes:
                        formatted_path.append(f"P{node}")
                    else:
                        formatted_path.append(str(node))

                SimLogger.log(f"starting with complete path: [{', '.join(formatted_path)}]",
                              order_id=self.current_order.order_id, component_type="AGV", component_id=self.agv_id)

                # Follow path and complete order
                for i, node in enumerate(self.path_to_follow):
                    self.agv_status = "moving"
                    self.travel_to_node_with_animation(node)

                    if node in pickup_nodes:
                        self.agv_status = "queued"
                        SimLogger.log(f"queued at node {node}",
                                      order_id=self.current_order.order_id, component_type="AGV",
                                      component_id=self.agv_id)
                        asrs = next((a for a in asrs_list if a.node_number == node), None)
                        if asrs:
                            wait_start = env.now()
                            asrs.enqueue_agv(self)
                            self.passivate()
                            wait_time = env.now() - wait_start
                            self.total_waiting_time += wait_time

                # Complete order
                self.current_order.mark_completed()
                SimLogger.log("completed order",
                              order_id=self.current_order.order_id, component_type="AGV", component_id=self.agv_id)

                # RESET ORDER TRACKING VARIABLES IMMEDIATELY AFTER ORDER COMPLETION
                self.current_order = None
                self.path_to_follow = []
                self.current_path_index = 0
                self.items_picked_up = 0
                self.total_items_in_order = 0

                # BATTERY CHECK AFTER ORDER COMPLETION (at drop-off)
                if self.battery < CHARGE_THRESHOLD:
                    SimLogger.log(f"battery low ({self.battery:.1f}%), going to charge after completing order",
                                  component_type="AGV", component_id=self.agv_id)
                    self.go_to_charger()  # This will charge and then return to depot
                else:
                    # Battery is sufficient - return to depot directly
                    self.return_to_depot()

                # Set status to idle (item counter will now show empty)
                self.agv_status = "idle"

    def return_to_depot(self):
        """Return AGV to depot from current position"""
        if self.current_node == DEPOT_NODE:
            return

        SimLogger.log(f"returning to depot from node {self.current_node}",
                      component_type="AGV", component_id=self.agv_id)

        if self.current_node == DROP_OFF_NODE:
            SimLogger.log(f"instant travel from drop-off to depot",
                          component_type="AGV", component_id=self.agv_id)
            self.current_node = DEPOT_NODE
            depot_x, depot_y = get_node_coordinates(DEPOT_NODE)
            self.animation_object.x = depot_x
            self.animation_object.y = depot_y
            SimLogger.log("arrived at depot",
                          component_type="AGV", component_id=self.agv_id)
            return

        path_to_depot = dijkstra_shortest_path(self.current_node, DEPOT_NODE, adjacency_matrix)

        for i, node in enumerate(path_to_depot):
            if i == 0:
                continue
            self.travel_to_node_with_animation(node)

        SimLogger.log("arrived at depot",
                      component_type="AGV", component_id=self.agv_id)

    def travel_to_node_with_animation(self, destination_node):
        """Travel from current node to destination node with smooth animation"""
        if self.current_node == destination_node:
            return

        dest_x, dest_y = get_node_coordinates(destination_node)

        if 1 <= destination_node <= 60:
            dest_y -= AnimationConfig.AGV_TRAVEL_OFFSET

        distance = calculate_distance_between_nodes(self.current_node, destination_node, adjacency_matrix)
        travel_time = distance / AGV_SPEED

        # ... rest of the method remains the same

        if travel_time <= 0:
            travel_time = 0.1

        order_id = self.current_order.order_id if self.current_order else None
        SimLogger.log(
            f"traveling from node {self.current_node} to node {destination_node} (distance: {distance}m, time: {travel_time:.1f}s)",
            order_id=order_id, component_type="AGV", component_id=self.agv_id)

        current_x, current_y = get_node_coordinates(self.current_node)

        if 1 <= self.current_node <= 60:
            current_y -= AnimationConfig.AGV_TRAVEL_OFFSET

        start_time = env.now()

        def interpolate_x(t):
            if travel_time <= 0:
                return dest_x
            elapsed_time = t - start_time
            progress = min(1.0, max(0.0, elapsed_time / travel_time))
            return current_x + (dest_x - current_x) * progress

        def interpolate_y(t):
            if travel_time <= 0:
                return dest_y
            elapsed_time = t - start_time
            progress = min(1.0, max(0.0, elapsed_time / travel_time))
            return current_y + (dest_y - current_y) * progress

        self.animation_object.x = interpolate_x
        self.animation_object.y = interpolate_y

        energy_consumed = distance * ENERGY_PER_METER

        if self.battery - energy_consumed < 0:
            SimLogger.log(f"Warning: Battery would go negative, setting to 0",
                          component_type="AGV", component_id=self.agv_id)
            self.battery = 0
        else:
            self.battery -= energy_consumed

        self.total_distance += distance
        self.hold(travel_time)
        self.current_node = destination_node

        final_x, final_y = get_node_coordinates(destination_node)

        if 1 <= destination_node <= 60:
            final_y -= AnimationConfig.AGV_TRAVEL_OFFSET

        self.animation_object.x = final_x
        self.animation_object.y = final_y

        order_id = self.current_order.order_id if self.current_order else None
        SimLogger.log(f"arrived at node {destination_node}",
                      order_id=order_id, component_type="AGV", component_id=self.agv_id)

    def loading_complete(self):
        if self.current_order and self.current_node in self.current_order.pickup_nodes:
            items_at_this_node = len(self.current_order.items_by_node.get(self.current_node, []))
            self.items_picked_up += items_at_this_node

            SimLogger.log(
                f"picked up {items_at_this_node} items, total: {self.items_picked_up}/{self.total_items_in_order}",
                order_id=self.current_order.order_id, component_type="AGV", component_id=self.agv_id)

        if 1 <= self.current_node <= 60:
            asrs_x, asrs_y = get_node_coordinates(self.current_node)
            self.animation_object.x = asrs_x
            self.animation_object.y = asrs_y - AnimationConfig.AGV_TRAVEL_OFFSET
            SimLogger.log(f"moved back to travel position after pickup",
                          component_type="AGV", component_id=self.agv_id)

        self.agv_status = "moving"
        self.activate()

    def go_to_charger(self):
        self.recharge_count += 1
        env.stats.total_recharges += 1
        self.agv_status = "moving"

        SimLogger.log(f"going to charger (battery: {self.battery:.1f}%)",
                      component_type="AGV", component_id=self.agv_id)

        if self.current_node == DROP_OFF_NODE:
            SimLogger.log(f"instant travel from drop-off to charger",
                          component_type="AGV", component_id=self.agv_id)
            charger_x, charger_y = get_node_coordinates(CHARGER_NODE)
            self.animation_object.x = charger_x
            self.animation_object.y = charger_y
        else:
            path_to_charger = dijkstra_shortest_path(self.current_node, DEPOT_NODE, adjacency_matrix)
            for i, node in enumerate(path_to_charger):
                if i == 0:
                    continue
                self.travel_to_node_with_animation(node)

        self.current_node = CHARGER_NODE
        self.agv_status = "charging"

        charge_time = (MAX_CHARGE - self.battery) / CHARGE_RATE
        self.hold(charge_time)

        self.battery = MAX_CHARGE
        SimLogger.log(f"finished charging, battery: {self.battery}%",
                      component_type="AGV", component_id=self.agv_id)

        self.current_node = DEPOT_NODE
        self.agv_status = "idle"
        depot_x, depot_y = get_node_coordinates(DEPOT_NODE)
        self.animation_object.x = depot_x
        self.animation_object.y = depot_y


# ===============================================================================
# AGV ANIMATION MANAGER - Controls how AGVs look and move on screen
# ===============================================================================

class AGVAnimationManager:
    """Handles all the visual aspects of an AGV - colors, position, battery display, etc."""
    def __init__(self, agv):
        self.agv = agv
        self.setup_animations()

    def setup_animations(self):
        """Create all visual elements for this AGV"""
        depot_x, depot_y = get_node_coordinates(DEPOT_NODE)

        # Main AGV rectangle
        self.agv.animation_object = sim.AnimateRectangle(
            spec=(-AnimationConfig.AGV_SIZE // 2, -AnimationConfig.AGV_SIZE // 2,
                  AnimationConfig.AGV_SIZE // 2, AnimationConfig.AGV_SIZE // 2),
            fillcolor=lambda t: self.get_agv_color(),
            linecolor="black",
            linewidth=2,
            x=depot_x, y=depot_y
        )

        # AGV ID number
        self.agv.animation_text = sim.AnimateText(
            text=lambda t: f"{self.agv.agv_id}",
            x=lambda t: self.agv.animation_object.x(t) if callable(
                self.agv.animation_object.x) else self.agv.animation_object.x,
            y=lambda t: self.agv.animation_object.y(t) if callable(
                self.agv.animation_object.y) else self.agv.animation_object.y,
            fontsize=AnimationConfig.FONT_SIZE,
            textcolor="black",
            text_anchor="c"
        )

        # Item counter (shows how many items picked up)
        self.agv.item_counter_text = sim.AnimateText(
            text=lambda t: self.get_item_counter_text(),
            x=lambda t: (self.agv.animation_object.x(t) if callable(
                self.agv.animation_object.x) else self.agv.animation_object.x) + 15,
            y=lambda t: (self.agv.animation_object.y(t) if callable(
                self.agv.animation_object.y) else self.agv.animation_object.y),
            fontsize=AnimationConfig.FONT_SIZE,
            textcolor="black",
            text_anchor="w"
        )

        # Battery level display
        self.setup_battery_animation()

    def setup_battery_animation(self):
        """Create the battery level bar under the AGV"""
        battery_width, battery_height = AnimationConfig.AGV_SIZE, 5

        # Red background (empty battery)
        self.agv.battery_bg = sim.AnimateRectangle(
            spec=(-battery_width // 2, -AnimationConfig.AGV_SIZE // 2 - battery_height - 2,
                  battery_width // 2, -AnimationConfig.AGV_SIZE // 2 - 2),
            fillcolor="red",
            linecolor="black", linewidth=1,
            x=lambda t: self.agv.animation_object.x(t) if callable(
                self.agv.animation_object.x) else self.agv.animation_object.x,
            y=lambda t: self.agv.animation_object.y(t) if callable(
                self.agv.animation_object.y) else self.agv.animation_object.y
        )

        # Green foreground (shows actual battery level)
        self.agv.battery_fg = sim.AnimateRectangle(
            spec=lambda t: self.get_battery_spec(battery_width, battery_height),
            fillcolor="green",
            linecolor="black", linewidth=1,
            x=lambda t: self.agv.animation_object.x(t) if callable(
                self.agv.animation_object.x) else self.agv.animation_object.x,
            y=lambda t: self.agv.animation_object.y(t) if callable(
                self.agv.animation_object.y) else self.agv.animation_object.y
        )

    def get_agv_color(self):
        """Return the right color based on what the AGV is doing"""
        status_colors = {
            "idle": AnimationConfig.AGV_IDLE,
            "moving": AnimationConfig.AGV_MOVING,
            "picking_up": AnimationConfig.AGV_PICKING,
            "queued": AnimationConfig.AGV_QUEUED,
            "charging": AnimationConfig.AGV_CHARGING
        }
        return status_colors.get(self.agv.agv_status, "gray")

    def get_item_counter_text(self):
        """Return the item counter text for display"""
        # Only show counter when AGV has an active order AND is not idle/charging
        if (self.agv.current_order is not None and
                self.agv.agv_status not in ["idle", "charging"]):
            return f"{self.agv.items_picked_up}/{self.agv.total_items_in_order}"
        else:
            return ""  # Empty string when idle, charging, or no order

    def get_battery_spec(self, battery_width, battery_height):
        """Calculate how much of the battery bar should be green"""
        battery_percentage = max(0, min(100, self.agv.battery)) / 100.0
        green_width = battery_width * battery_percentage

        return (-battery_width // 2, -AnimationConfig.AGV_SIZE // 2 - battery_height - 2,
                -battery_width // 2 + green_width, -AnimationConfig.AGV_SIZE // 2 - 2)

# ===============================================================================
# ASRS ANIMATION MANAGER - Controls how ASRS machines look on screen
# ===============================================================================

class ASRSAnimationManager:
    """Handles visual display for ASRS machines - shows when they're working and container status"""
    def __init__(self, asrs):
        self.asrs = asrs
        self.setup_animations()

    def setup_animations(self):
        """Create visual elements for this ASRS machine"""
        asrs_x, asrs_y = get_node_coordinates(self.asrs.node_number)

        # Status text showing what the ASRS is doing
        self.asrs.status_text = sim.AnimateText(
            text=lambda t: self.get_picking_status_text(),
            x=asrs_x - 50,
            y=asrs_y,
            fontsize=AnimationConfig.FONT_SIZE,
            textcolor="darkorchid",
            text_anchor="e"
        )

        # Container indicator (only if using new innovation method)
        if INNOVATION:
            container_width, container_height = AnimationConfig.NODE_RADIUS * 2, 4
            self.asrs.container_indicator = sim.AnimateRectangle(
                spec=(-container_width // 2, -AnimationConfig.NODE_RADIUS - container_height - 3,
                      container_width // 2, -AnimationConfig.NODE_RADIUS - 3),
                fillcolor=lambda t: self.get_container_color(),
                linecolor="black", linewidth=1,
                x=asrs_x, y=asrs_y
            )
        else:
            # No container indicator for old method
            self.asrs.container_indicator = None

    def get_picking_status_text(self):
        """Return text showing what the ASRS is currently picking"""
        if self.asrs.current_picking_order and self.asrs.current_picking_items > 0:
            return f"ASRS picking {self.asrs.current_picking_items} for O-{self.asrs.current_picking_order}"
        else:
            return ""

    def get_container_color(self):
        """Return color showing if container has items ready"""
        if INNOVATION and hasattr(self.asrs, 'container') and self.asrs.container.has_items():
            return AnimationConfig.CONTAINER_READY
        else:
            return AnimationConfig.CONTAINER_EMPTY

# ===============================================================================
# LOGGING SYSTEM - Prints messages about what's happening
# ===============================================================================

class SimLogger:
    """Simple system to print organized messages about what's happening in the simulation"""
    @staticmethod
    def log(message, order_id=None, component_type=None, component_id=None):
        """Print a message with time stamp and component info"""
        time_str = f"[{env.now():.1f}T]"

        if order_id:
            order_str = f"O{order_id} - "
        else:
            order_str = ""

        if component_type and component_id:
            comp_str = f"{component_type} {component_id}: "
        else:
            comp_str = ""

        print(f"{time_str}: {order_str}{comp_str}{message}")

# ===============================================================================
# STATISTICS TRACKING - Keeps track of simulation performance
# ===============================================================================

class WarehouseStats:
    def __init__(self):
        self.orders_created = 0
        self.orders_completed = 0
        self.total_order_time = 0.0
        self.order_times = []
        self.total_recharges = 0
        self.container_ready_transfers = 0
        self.container_wait_transfers = 0
        self.sequential_transfers = 0

    def update_agv_stats(self, agvs):
        self.total_agv_distance = sum(agv.total_distance for agv in agvs)
        self.total_agv_waiting_time = sum(agv.total_waiting_time for agv in agvs)

    def print_stats(self):
        print("\n===== Enhanced Warehouse Simulation Statistics =====")
        print(f"Innovation mode: {'ON' if INNOVATION else 'OFF'}")
        print(f"Simulation time: {env.now() / 60:.2f} minutes ({env.now() / 3600:.2f} hours)")
        print(f"Orders created: {self.orders_created}")
        print(f"Orders completed: {self.orders_completed}")

        if self.orders_completed > 0:
            avg_order_time = self.total_order_time / self.orders_completed / 60
            print(f"Average order completion time: {avg_order_time:.2f} minutes")

            throughput = self.orders_completed / (env.now() / 3600)
            print(f"Throughput: {throughput:.2f} orders/hour")

            if self.order_times:
                min_time = min(self.order_times) / 60
                max_time = max(self.order_times) / 60
                print(f"Minimum order time: {min_time:.2f} minutes")
                print(f"Maximum order time: {max_time:.2f} minutes")

        self.update_agv_stats(agv_list)
        print(f"Total AGV distance traveled: {self.total_agv_distance:.2f} meters")
        print(f"Total AGV waiting time: {self.total_agv_waiting_time / 60:.2f} minutes")
        print(f"Total battery recharges: {self.total_recharges}")
        if agv_list:
            print(f"Average AGV battery level: {sum(agv.battery for agv in agv_list) / len(agv_list):.1f}%")

        print(f"\n===== Innovation Performance Metrics =====")
        total_transfers = self.container_ready_transfers + self.container_wait_transfers + self.sequential_transfers
        if total_transfers > 0:
            if INNOVATION:
                ready_percentage = (self.container_ready_transfers / total_transfers) * 100
                wait_percentage = (self.container_wait_transfers / total_transfers) * 100
                print(f"Container ready on arrival: {self.container_ready_transfers} ({ready_percentage:.1f}%)")
                print(f"AGV waited for container: {self.container_wait_transfers} ({wait_percentage:.1f}%)")
                print(f"Innovation effectiveness: {ready_percentage:.1f}% of transfers benefited from pre-preparation")
            else:
                print(f"Sequential transfers: {self.sequential_transfers} (100.0%)")
                print("Innovation mode OFF - all transfers use traditional sequential processing")

        print(f"\n===== Individual AGV Statistics =====")
        for agv in agv_list:
            print(f"AGV {agv.agv_id}:")
            print(f"  - Total recharges: {agv.recharge_count}")
            print(f"  - Final battery level: {agv.battery:.1f}%")
            print(f"  - Total distance traveled: {agv.total_distance:.2f} meters")
            print(f"  - Total waiting time: {agv.total_waiting_time / 60:.2f} minutes")
            print(f"  - Current status: {agv.agv_status}")
            print(f"  - Current node: {agv.current_node}")


# Initialize simulation
env = sim.Environment(trace=False)
#env.animation_parameters(animate=True, width=AnimationConfig.WINDOW_WIDTH, height=AnimationConfig.WINDOW_HEIGHT)
env.animation_parameters(animate=False)
env.stats = WarehouseStats()

# Load adjacency matrix and warehouse data
adjacency_matrix = load_adjacency_matrix()
warehouse_items = load_warehouse_data()

# Initialize component lists and queues
asrs_list = []
agv_list = []
all_orders = []
available_agvs = []
order_queue = sim.Queue("Order_Queue")

# Setup animation
setup_animation()

# Create ASRS systems for each unique node that has items
unique_nodes = set(item['AisleLocation'] for item in warehouse_items)
for node in sorted(unique_nodes):
    asrs = ASRS(name=f"ASRS_Node_{node}", node_number=node)
    asrs_list.append(asrs)
    warehouse_system.register_asrs(node, asrs)

SimLogger.log(f"Created {len(asrs_list)} ASRS systems for nodes: {sorted(unique_nodes)}")

# Create AGVs
num_agv=AGV_NUMBR
for i in range(num_agv):
    agv = AGV(name=f"AGV_{i + 1}", agv_id=i + 1)
    agv_list.append(agv)
    available_agvs.append(agv)
    warehouse_system.register_agv(agv)

# Create dispatcher
dispatcher = Dispatcher(name="Dispatcher")
warehouse_system.register_dispatcher(dispatcher)

# Create order generator
order_definitions = generate_orders_ahead(SEED, SIM_DURATION, warehouse_items)
order_feeder = OrderFeeder(name="OrderFeeder", order_definitions=order_definitions)


# Run simulation
SimLogger.log(f"Starting enhanced simulation with innovation mode {'ON' if INNOVATION else 'OFF'}")
SimLogger.log("AGVs will now show live movement with color coding:")
SimLogger.log("- Green: Idle AGVs")
SimLogger.log("- Orange: Moving/Queued AGVs")
SimLogger.log("- Red: AGVs actively picking up at ASRS")
SimLogger.log("- Blue: AGVs charging")


# # Run simulation for 1 hour
# env.run(till=3600)

# Run simulation for whole duration with 1 hour progress intervalls
progress_interval = 3600  # 1 hour intervals
for progress in range(progress_interval, SIM_DURATION + progress_interval, progress_interval):
    target_time = min(progress, SIM_DURATION)
    env.run(till=target_time)
    print(f"Simulated {target_time / 3600:.1f} hours ({target_time / SIM_DURATION * 100:.1f}%)")

# Print final statistics
env.stats.print_stats()

# Print additional node-based statistics
print(f"\nNode-based Navigation Summary:")
print(f"Total unique pickup nodes: {len(unique_nodes)}")
if agv_list:
    avg_distance_per_agv = sum(agv.total_distance for agv in agv_list) / len(agv_list)
    print(f"Average distance per AGV: {avg_distance_per_agv:.2f} meters")

# Summary of total orders and total items
total_orders = len(all_orders)
total_items = sum(len(order.items) for order in all_orders)

print(f"\n📦 Summary:")
print(f"Total orders created: {total_orders}")
print(f"Total items across all orders: {total_items}")
print(f"Average items per order: {total_items / total_orders:.2f}" if total_orders > 0 else "No orders created.")


