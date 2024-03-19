import math
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class Dijkstra:
    def __init__(self, obstacle_map, resolution, robot_radius):
        self.obstacle_map = obstacle_map
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.motion = self.get_motion_model()
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = obstacle_map.shape
        self.x_width, self.y_width = self.max_x, self.max_y

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return f"{self.x},{self.y},{self.cost},{self.parent_index}"

    def planning(self, sx, sy, gx, gy):
        print("Start path planning...")
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)
        print("Start and goal nodes initialized.")

        open_set, closed_set = dict(), dict()
        open_set[self.calc_index(start_node)] = start_node
        print("Start node added to open set.")

        while True:
            if not open_set:
                break  # Exit loop if open set is empty
            c_id = min(open_set, key=lambda o: open_set[o].cost)
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                print("Goal reached.")
                goal_node.cost = current.cost
                goal_node.parent_index = current.parent_index
                break

            del open_set[c_id]
            closed_set[c_id] = current

            for move_x, move_y, move_cost in self.motion:
                node = self.Node(current.x + move_x,
                                 current.y + move_y,
                                 current.cost + move_cost, c_id)
                n_id = self.calc_index(node)

                if n_id in closed_set or not self.verify_node(node):
                    continue

                if n_id not in open_set or node.cost < open_set[n_id].cost:
                    open_set[n_id] = node

        print("Path planning completed.")
        rx, ry = self.calc_final_path(goal_node, closed_set)
        return rx, ry





    # Robot movement model, able to move to any of the 8 surrounding grid cells in each step
    @staticmethod
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

    def calc_position(self, index, minp):
        pos = minp + index
        return pos


    def calc_xy_index(self, position, minp):
        return round(position - minp)

    def calc_index(self, node):
        return node.y * self.x_width + node.x

    # def verify_node(self, node):
    #     if node.x < 0 or node.y < 0 or node.x >= self.x_width or node.y >= self.y_width:
    #         return False
    #     if self.obstacle_map[node.y, node.x]:
    #         return False
    #     return True

    def verify_node(self, node):
        if node.x < 0 or node.y < 0 or node.x >= self.x_width or node.y >= self.y_width:
            return False

        # 计算机器人需要的最小空闲区域半径，以格子数为单位
        clearance_required = math.ceil(0.3 / self.resolution)  # 0.3m转弯半径对应的格子数

        # 检查机器人周围是否有足够的空间
        for dx in range(-clearance_required, clearance_required + 1):
            for dy in range(-clearance_required, clearance_required + 1):
                # 计算检查点的坐标
                check_x = node.x + dx
                check_y = node.y + dy

                # 确保检查点在地图内
                if check_x < 0 or check_y < 0 or check_x >= self.x_width or check_y >= self.y_width:
                    continue  # 如果检查点超出地图范围，则跳过

                # 如果检查点是障碍物，则当前节点不满足条件
                if self.obstacle_map[check_y, check_x]:
                    return False

        return True

    def calc_final_path(self, goal_node, closed_set):
        rx = [self.calc_position(goal_node.x, self.min_x)]
        ry = [self.calc_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_position(n.x, self.min_x))
            ry.append(self.calc_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry



    def calculate_total_distance(self, rx, ry):
        """
        Calculate the total distance of the path based on the resolution.
        """
        total_distance = 0.0
        for i in range(1, len(rx)):
            dx = rx[i] - rx[i - 1]
            dy = ry[i] - ry[i - 1]
            distance = math.sqrt(dx ** 2 + dy ** 2) * self.resolution
            total_distance += distance
        return total_distance


def main():

    print("Starting main program...")

    obstacle_map = np.load("../map/obstacle_map.npy").astype(np.bool_)

    sx, sy = 345, 95  # 起点
    gx, gy = 20, 705  # 终点
    grid_size = 0.2  # 网格大小
    robot_radius = 0.3  # 机器人半径，根据障碍物密度调整

    if True:
        # plt.plot(ox, oy, '.k')  # 障碍物黑色
        plt.imshow(obstacle_map, cmap='gray', origin='upper')
        plt.plot(sx, sy, 'og')  # 起点绿色
        plt.plot(gx, gy, 'xb')  # 终点蓝色
        plt.grid(True)
        plt.axis('equal')  # 坐标轴刻度间距等长
        plt.show()

    dijkstra = Dijkstra(obstacle_map, grid_size, robot_radius)
    rx, ry = dijkstra.planning(sx, sy, gx, gy)

    # Calculate and print the total distance
    total_distance = dijkstra.calculate_total_distance(rx, ry)
    print(f"Total distance of the path: {total_distance} meters")

    obstacle_map = np.logical_not(obstacle_map)
    if True:
        plt.imshow(obstacle_map, cmap='gray', origin='upper')
        plt.plot(sx, sy, "og")  # 起点绿色
        plt.plot(gx, gy, "xb")  # 终点蓝色
        plt.plot(rx, ry, "-r")  # 路径红色
        plt.grid(True)
        plt.axis("equal")
        plt.show()

if __name__ == '__main__':
    main()
