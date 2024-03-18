import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.spatial.distance import euclidean


def dilate_obstacles(map, radius):
    # Round the radius to the nearest integer, as we need integers to use in the range function
    radius = int(round(radius * 2))  # Adjust the radius to be in terms of grid units, then round it
    # Create a structural element that simulates a circle of the given radius
    struct = np.array([[euclidean((x - radius, y - radius), (0, 0)) <= radius for x in range(2 * radius + 1)] for y in range(2 * radius + 1)])
    return binary_dilation(map == 0, structure=struct).astype(map.dtype)

def get_path(previous, current):
    path = [current]
    distance = 0.0
    while any(previous[current[1], current[0], :]):
        next_node = previous[current[1], current[0], :].astype(int)
        if np.any(next_node != current):
            step_distance = euclidean(current, next_node) * 0.2  # Each grid represents 0.2 meters
            distance += step_distance
            current = next_node
            path.insert(0, current)
    return np.array(path), distance


def heuristic(node, goal, resolution=0.2):
    """
    Heuristic function, using Euclidean distance for calculation.
    :param node: Current node coordinates.
    :param goal: Target node coordinates.
    :param resolution: Map resolution.
    """
    dis = euclidean(node, goal) * resolution
    print(f"Heuristic calculated for node {node} to goal {goal}: {dis}")
    return dis

def get_motion_model():
    # [dx, dy, cost]
    motion = [[1, 0, 1],  # Right
              [0, 1, 1],  # Up
              [-1, 0, 1],  # Left
              [0, -1, 1],  # Down
              [-1, -1, math.sqrt(2)],  # Down-Left
              [-1, 1, math.sqrt(2)],  # Up-Left
              [1, -1, math.sqrt(2)],  # Down-Right
              [1, 1, math.sqrt(2)]]  # Up-Right
    return motion


def astar(map, start, goal):
    print("A* Algorithm Starts")
    motion = get_motion_model()  # Get the motion model and cost using get_motion_model

    open_set = [start]
    closed_set = []
    g_score = np.inf * np.ones_like(map, dtype=np.float64)
    g_score[start[1], start[0]] = 0
    f_score = np.inf * np.ones_like(map, dtype=np.float64)
    f_score[start[1], start[0]] = heuristic(start, goal)
    previous = np.zeros((*map.shape, 2), dtype=int)

    while open_set:
        current = min(open_set, key=lambda x: f_score[x[1], x[0]])
        print(f"Current node: {current}")
        if current == goal:
            print("Goal reached")
            path, distance = get_path(previous, current)
            return path, distance
        open_set.remove(current)
        closed_set.append(current)

        for move in motion:
            dx, dy, cost = move
            neighbor = [current[0] + dx, current[1] + dy]
            if not (0 <= neighbor[1] < map.shape[0] and 0 <= neighbor[0] < map.shape[1]):
                continue
            if map[neighbor[1], neighbor[0]] == 1 or neighbor in closed_set:
                continue
            tentative_g_score = g_score[current[1], current[0]] + cost  # Using the cost of movement
            if tentative_g_score < g_score[neighbor[1], neighbor[0]]:
                previous[neighbor[1], neighbor[0]] = current
                g_score[neighbor[1], neighbor[0]] = tentative_g_score
                f_score[neighbor[1], neighbor[0]] = tentative_g_score + heuristic(neighbor, goal)
                if neighbor not in open_set:
                    open_set.append(neighbor)
    print("No path found")
    return [], 0


def visualize_map_and_path(map, path):
    plt.figure()
    plt.imshow(1 - map, cmap='gray', origin='upper')
    plt.plot(path[:, 0], path[:, 1], 'r-', linewidth=2)
    plt.plot(path[0, 0], path[0, 1], 'go', markersize=10, markerfacecolor='g')
    plt.plot(path[-1, 0], path[-1, 1], 'ro', markersize=10, markerfacecolor='r')
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Map and Path')
    plt.show()

def find_and_visualize_path(map, locations):
    # Extract the coordinates of the start and goal locations
    start = locations.get('start')
    goal = locations.get('store')  # Or other target location keywords
    if not start or not goal:
        print("Start or goal location not provided in locations dictionary.")
        return

    # Find the path using the A* algorithm
    path, distance = astar(map, start, goal)

    # If a path is found, print the distance and visualize it
    if path.size > 0:
        print(f"Path from {start} to {goal}: Distance = {distance} meters")  # Print the distance
        visualize_map_and_path(map, path)
    else:
        print("No path found from start to goal.")

def visualize_map_and_multiple_paths(map, paths):
    plt.figure()
    plt.imshow(1 - map, cmap='gray', origin='upper')
    for path in paths:
        plt.plot(path[:, 0], path[:, 1], '-', linewidth=2)
        plt.plot(path[0, 0], path[0, 1], 'go', markersize=10, markerfacecolor='g')
        plt.plot(path[-1, 0], path[-1, 1], 'ro', markersize=10, markerfacecolor='r')
    plt.axis('equal')
    plt.grid(True)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Map and Multiple Paths')
    plt.show()

def find_and_visualize_all_paths(map, locations):
    total_distance = 0.0
    paths = []
    distances = {}
    for start_key, start in locations.items():
        for goal_key, goal in locations.items():
            if start_key != goal_key:
                path, distance = astar(map, start, goal)
                if path.size > 0:
                    paths.append(path)
                    distances[(start_key, goal_key)] = distance
                    total_distance += distance
    # most_efficient_path, most_efficient_distance = solve_tsp(distances)
    print(f"Total distance: {total_distance} meters")
    # print(f"Most efficient path: {most_efficient_path} with distance: {most_efficient_distance} meters")
    visualize_map_and_multiple_paths(map, paths)


def solve_tsp(distances):
    # Assume starting point is 'start'
    start = 'start'
    # Initialize the set of visited points, including the starting point
    visited = {start}
    # Initialize the current position
    current = start
    # Initialize the total distance
    total_distance = 0
    # Initialize the visit order
    visit_order = [start]

    # Continue to find the next closest point while there are unvisited points
    while len(visited) < len(distances) // 2 + 1:  # The total number of points is twice the number of point pairs plus one
        # Find the closest point among the remaining unvisited points
        next_point, min_distance = None, float('inf')
        for (start_point, end_point), distance in distances.items():
            if start_point == current and end_point not in visited and distance < min_distance:
                next_point, min_distance = end_point, distance
            elif end_point == current and start_point not in visited and distance < min_distance:
                next_point, min_distance = start_point, distance

        # Update the list of visited and the total distance
        if next_point:
            visited.add(next_point)
            total_distance += min_distance
            visit_order.append(next_point)
            current = next_point

    # Distance back to the start
    if visit_order[-1] != start:
        total_distance += distances.get((current, start), 0)
        visit_order.append(start)

    return visit_order, total_distance


image_path = "../map/vivocity_freespace.png"
img = Image.open(image_path).convert('L')
obstacle_map = np.array(img) == 0
obstacle_map_uint8 = obstacle_map.astype(np.uint8)
np.save("../map/obstacle_map.npy", obstacle_map_uint8)
map = np.load("../map/obstacle_map.npy")
print("Map loaded")
map_dilated = dilate_obstacles(map, radius=1.5)  # Each grid is 0.2m, so a radius of 0.3m is approximately 1.5 grids
locations = {
    'start': [345, 95],# Start from the level 2 Escalator
    # 'snacks': [470, 475],  # Garrett Popcorn
    'store':  [20, 705],    # DJI Store
    # 'movie':  [940, 545],   # Golden Village
    # 'food':   [535, 800],   # PUTIEN
}
# find_and_visualize_all_paths(map, locations)
find_and_visualize_path(map, locations)
