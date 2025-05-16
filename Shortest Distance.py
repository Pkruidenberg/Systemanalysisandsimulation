import networkx as nx
import matplotlib.pyplot as plt
import heapq
import pandas as pd
import numpy as np
import random

random.seed(42)


def create_dict(existing_dict):
    for key in range(61):
        existing_dict[key] = None  # Or your desired default value
    return existing_dict

def snake_pattern(empty_dict,between_aisle=3, start_end=5, inside_aisle=25):
    graph = {}
    for key in range(61):  # Keys 0-62
        next_key = key + 1
        if next_key > 61:
            graph[key] = {}
        elif key == 0 or key == 60:
            graph[key] = {next_key: start_end}
        else:
            # Determine pattern position using (key-1) % 3
            match (key - 1) % 3:
                case 0 | 1:
                    graph[key] = {next_key: inside_aisle}
                case 2:
                    graph[key] = {next_key: between_aisle}
    return graph

def add_shortcuts(graph, between_aisle=3, shortcut5_distance=5):
    """Adds shortcut connections to the graph with specified distances.
    
    Args:
        graph: The dictionary to modify
        default_shortcut: Distance for regular shortcuts (default: 3)
        shortcut5_distance: Distance for short_cut_5 connections (default: 5)
    """
    # Define all shortcut groups
    shortcut_groups = [
        ([(1,6), (7,12), (13,18), (19,24), (25,30)], between_aisle),
        ([(4,9), (10,15), (16,21), (22,27)], between_aisle),
        ([(31,36), (37,42), (43,48), (49,54), (55,60)], between_aisle),
        ([(34,39), (40,45), (46,51), (52,57)], between_aisle),
        ([(6,55), (12,49), (18,43), (24,37), (30,31)], shortcut5_distance)
    ]

    # Add all connections to the graph
    for connections, distance in shortcut_groups:
        for a, b in connections:
            # Create empty dict if key doesn't exist (shouldn't happen in original graph)
            graph.setdefault(a, {})
            graph[a][b] = distance

    return graph

def plot_warehouse_from_matrix(graph_dict, visited_nodes=None, requested_nodes=None):

    G = nx.DiGraph()
    
    for node, neighbors in graph_dict.items():
        for neighbor, weight in neighbors.items():
            G.add_edge(node, neighbor, weight=weight)

    pos = {}
    matrix = [
        [33, 32, 31, None, 30, 29, 28],
        [34, 35, 36, None, 25, 26, 27],
        [39, 38, 37, None, 24, 23, 22],
        [40, 41, 42, None, 19, 20, 21],
        [45, 44, 43, None, 18, 17, 16],
        [46, 47, 48, None, 13, 14, 15],
        [51, 50, 49, None, 12, 11, 10],
        [52, 53, 54, None, 7, 8, 9],
        [57, 56, 55, None, 6, 5, 4],
        [58, 59, 60, None, 1, 2, 3],
    ]

    for row_idx, row in enumerate(matrix):
        for col_idx, node in enumerate(row):
            if node is not None:
                x = col_idx
                y = len(matrix) - row_idx - 1
                pos[node] = (x, y)

    pos[61] = (2, -1)
    pos[0] = (4, -1)

    # Node coloring
    node_colors = []
    for n in G.nodes:
        if requested_nodes and n in requested_nodes:
            node_colors.append('green')
        elif visited_nodes and n in visited_nodes:
            node_colors.append('orange')
        else:
            node_colors.append('lightblue')

    # Highlight visited edges
    path_edges = []
    if visited_nodes:
        for i in range(len(visited_nodes) - 1):
            if G.has_edge(visited_nodes[i], visited_nodes[i+1]):
                path_edges.append((visited_nodes[i], visited_nodes[i+1]))

    plt.figure(figsize=(15, 10))
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=400)
    nx.draw_networkx_labels(G, pos, font_size=8)

    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20, connectionstyle='arc3,rad=0.1')
    if path_edges:
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', arrows=True, arrowsize=20, connectionstyle='arc3,rad=0.1', width=2)

    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Warehouse Layout Visualization with Path and Request Highlights", pad=20)
    plt.grid(True)
    plt.show()


# Example usage:
empty_dict = {}
create_dict(empty_dict)

#parameters to be adjusted
between_aisle=4
start_end=10
between_the_2_aisle_columns = 8 
inside_aisle=25

#this is an example of the nodes that need to be visited for the completion of a given order.
node_sequence = [0, 5, 5, 11, 32, 56, 61]

snake_dictionary= snake_pattern(empty_dict, between_aisle,start_end,inside_aisle)
modified_graph = add_shortcuts(snake_dictionary,between_aisle,between_the_2_aisle_columns)

#Print statement if you want to check it more.
'''
if modified_graph is not None:
    # Print in a readable format
    for node, neighbors in sorted(modified_graph.items()):
        print(f"{node}: {neighbors}")
else:
    print("Error: modified_graph is None! Check your functions' return statements")
'''

#This is the geenral case. Best to keep it just in case we decide to addapt something
def compute_all_shortest_paths(modified_graph):
    nodes = list(modified_graph.keys())
    all_distances = {}

    for start in nodes:
        distances_from_start = {}
        visited = set()
        heap = [(0, start)]

        while heap:
            curr_dist, curr_node = heapq.heappop(heap)
            if curr_node in visited:
                continue
            visited.add(curr_node)

            if curr_node > start:
                distances_from_start[curr_node] = curr_dist

            for neighbor, dist in modified_graph.get(curr_node, {}).items():
                if neighbor not in visited:
                    heapq.heappush(heap, (curr_dist + dist, neighbor))

        # Sort by node number
        sorted_distances = dict(sorted(distances_from_start.items()))
        all_distances[start] = sorted_distances
        print(f'-- Shortest distances from {start} to higher-indexed nodes --')
        print(sorted_distances)

    return all_distances

def compute_sequence_path_distance(modified_graph, node_sequence):
    if 0 not in node_sequence or 61 not in node_sequence:
        raise ValueError("The node sequence must include start node 0 and end node 61.")
    
    #Absolutely no idea why I had it in the first place
    #if (len(node_sequence) - 2) % 2 != 0:
    #    raise ValueError(f"The number of intermediate nodes ({len(node_sequence) - 2}) must be divisible by 2.")

    total_distance = 0
    full_path = []

    for i in range(len(node_sequence) - 1):
        start = node_sequence[i]
        end = node_sequence[i + 1]

        # Dijkstra with path tracking
        heap = [(0, start, [start])]
        visited = set()

        found = False
        while heap:
            curr_dist, curr_node, path = heapq.heappop(heap)

            if curr_node == end:
                total_distance += curr_dist
                # Avoid duplicating the start node
                if full_path and full_path[-1] == path[0]:
                    full_path.extend(path[1:])
                else:
                    full_path.extend(path)
                found = True
                break

            if curr_node in visited:
                continue
            visited.add(curr_node)

            for neighbor, dist in modified_graph.get(curr_node, {}).items():
                if neighbor not in visited:
                    heapq.heappush(heap, (curr_dist + dist, neighbor, path + [neighbor]))

        if not found:
            raise ValueError(f"No path found from {start} to {end}")

    #print(f"Total distance for visiting nodes {node_sequence} in order: {total_distance}")
    return total_distance, full_path

"""
this is a test try, to see if the function behaves correctly.
If you have multiple items from the same node [0, 17, 17, 38, 41, 59, 59, 61] the function still works just fine
"""
#total_dist, visited_nodes = compute_sequence_path_distance(modified_graph, node_sequence)
#plot_warehouse_from_matrix(modified_graph)
#plot_warehouse_from_matrix(modified_graph, visited_nodes=visited_nodes, requested_nodes=node_sequence)

#############################################################################################################################
##################################################  Reading the CSV file created via "Generate_item_dataset" with pandas and doing some statistics
##################################################  on the order travelled distance
#############################################################################################################################


file_path = r"C:\Users\MSI\Desktop\downloaded files from uni\Year 4\Q3\System Analysis and Simulation\generated_orders_24h.csv"
df = pd.read_csv(file_path)

# Group by TimestampMin and aggregate AisleLocation into a list
# Step 1: Group and sort AisleLocation
grouped_df = df.groupby('TimestampMin')['AisleLocation'].apply(
    lambda x: [0] + sorted(map(int, x)) + [61]
).reset_index()

# Calculate total distance and store visited path
def calculate_distance_and_path(row):
    node_sequence = row['AisleLocation']
    total_dist, visited_nodes = compute_sequence_path_distance(modified_graph, node_sequence)
    return pd.Series({'TotalDistance': total_dist, 'VisitedNodes': visited_nodes})

# Step 3: Apply function to each row
grouped_df[['TotalDistance', 'VisitedNodes']] = grouped_df.apply(calculate_distance_and_path, axis=1)

# Calculate the number of nodes visited per order
grouped_df['NumVisitedNodes'] = grouped_df['VisitedNodes'].apply(len)

# Compute statistics
mean_nodes = grouped_df['NumVisitedNodes'].mean()
median_nodes = grouped_df['NumVisitedNodes'].median()
std_nodes = grouped_df['NumVisitedNodes'].std()

print(f"Mean number of nodes visited: {mean_nodes:.2f}")
print(f"Median number of nodes visited: {median_nodes}")
print(f"Standard deviation of nodes visited: {std_nodes:.2f}")

print(grouped_df['TotalDistance'].describe())
print(grouped_df['NumVisitedNodes'].describe())


# Additional stats
median_distance = grouped_df['TotalDistance'].median()
print(f"Median: {median_distance:.2f}")


'''
#REMOVED, SINCE WE MIGHT NOT NEED IT, ESPECIALLY IN THE REPORT, BUT IT WAS INTERESTING TO SEE IT

# Step 5: Select 3 orders (the shortest, longest and random)
shortest_order = grouped_df.loc[grouped_df['TotalDistance'].idxmin()]
longest_order = grouped_df.loc[grouped_df['TotalDistance'].idxmax()]
random_order = grouped_df.sample(n=1).iloc[0]

# Step 6: Visualize paths
plot_warehouse_from_matrix(modified_graph, visited_nodes=shortest_order['VisitedNodes'], requested_nodes=shortest_order['AisleLocation'])
plot_warehouse_from_matrix(modified_graph, visited_nodes=random_order['VisitedNodes'], requested_nodes=random_order['AisleLocation'])
plot_warehouse_from_matrix(modified_graph, visited_nodes=longest_order['VisitedNodes'], requested_nodes=longest_order['AisleLocation'])
'''

# Extract data
distances = grouped_df['TotalDistance']

# Compute statistics
mean_val = distances.mean()
median_val = distances.median()

# Plot histogram and capture bin data
plt.figure(figsize=(10, 6))
counts, bins, patches = plt.hist(distances, bins=20, color='skyblue', edgecolor='black', alpha=0.8)

# Annotate each bar
for count, bin_left, patch in zip(counts, bins[:-1], patches):
    plt.text(bin_left + (bins[1] - bins[0]) / 2, count + 1,  # Add slight offset above bar
             str(int(count)), ha='center', va='bottom', fontsize=9)

# Add mean and median lines
plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
plt.axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')

# Labels and title
plt.title('Histogram of Total Travel Distance per Order (with Counts)', fontsize=14)
plt.xlabel('Total Distance', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
