import math
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


min_set = 10
show_animation = True

class A_Start:
    # 初始化
    def __init__(self, obstacle_map, resolution, robot_radius):
        self.obstacle_map = obstacle_map
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.motion = self.get_motion_model()
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = obstacle_map.shape
        self.x_width, self.y_width = self.max_x, self.max_y

    # 构建节点，每个网格代表一个节点
    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # 网格索引
            self.y = y
            self.cost = cost  # 路径值
            self.parent_index = parent_index  # 该网格的父节点

        def __str__(self):
            return str(self.x) + ',' + str(self.y) + ',' + str(self.cost) + ',' + str(self.parent_index)

    # 寻找最优路径，网格起始坐标(sx,sy)，终点坐标（gx,gy）
    def planning(self, sx, sy, gx, gy):
        print("开始规划路径...")
        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)
        print("起点和终点初始化完成。")

        open_set, closed_set = dict(), dict()
        open_set[self.calc_index(start_node)] = start_node
        print("起点加入开放列表。")

        while 1:
            print("开始从开放列表中选择下一个节点...")
            c_id = min(open_set,
                       key=lambda o: open_set[o].cost + \
                                     self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                print("到达终点。")
                goal_node.cost = current.cost
                goal_node.parent_index = current.parent_index
                break

            del open_set[c_id]
            closed_set[c_id] = current
            print(f"节点 {c_id} 从开放列表移动到关闭列表。")

            for move_x, move_y, move_cost in self.motion:
                print("检查邻接节点...")
                node = self.Node(current.x + move_x,
                                 current.y + move_y,
                                 current.cost + move_cost, c_id)
                n_id = self.calc_index(node)

                if n_id in closed_set:
                    continue
                if not self.verify_node(node):
                    continue
                if n_id not in open_set:
                    open_set[n_id] = node
                    print(f"邻接节点 {n_id} 添加到开放列表。")
                else:
                    if node.cost <= open_set[n_id].cost:
                        open_set[n_id] = node
                        print(f"更新开放列表中的节点 {n_id}。")

        print("路径规划完成。")
        rx, ry = self.calc_final_path(goal_node, closed_set)
        return rx, ry

    # ------------------------------ #
    # A* 的启发函数
    # ------------------------------ #
    @staticmethod
    def calc_heuristic(n1, n2):
        w = 1.0  # 单个启发函数的权重，如果有多个启发函数，权重可以设置的不一样
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)  # 当前网格和终点距离
        return d

    # 机器人行走的方式，每次能向周围移动8个网格移动
    @staticmethod
    def get_motion_model():
        # [dx, dy, cost]
        motion = [[1, 0, 1],  # 右
                  [0, 1, 1],  # 上
                  [-1, 0, 1],  # 左
                  [0, -1, 1],  # 下
                  [-1, -1, math.sqrt(2)],  # 左下
                  [-1, 1, math.sqrt(2)],  # 左上
                  [1, -1, math.sqrt(2)],  # 右下
                  [1, 1, math.sqrt(2)]]  # 右上
        return motion

    # 绘制栅格地图
    def calc_obstacle_map(self, ox, oy):
        print("开始计算障碍物地图...")
        self.min_x = min(ox)
        self.min_y = min(oy)
        self.max_x = max(ox)
        self.max_y = max(oy)

        self.x_width = round((self.max_x - self.min_x) // self.resolution)
        self.y_width = round((self.max_y - self.min_y) // self.resolution)
        print("地图边界和网格尺寸计算完成。")

        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        print("初始化障碍物地图完成。")

        for ix in range(self.x_width):
            x = self.calc_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.robot_radius:
                        self.obstacle_map[ix][iy] = True
                        print(f"设置障碍物在网格 ({ix}, {iy})。")
                        break

        print("障碍物地图计算完成。")

    # 根据网格编号计算实际坐标
    def calc_position(self, index, minp):
        # minp代表起点坐标，左下x或左下y
        pos = minp + index * self.resolution  # 网格点左下左下坐标
        return pos

    # 位置坐标转为网格坐标
    def calc_xy_index(self, position, minp):
        # (目标位置坐标-起点坐标)/一个网格的长度==>目标位置的网格索引
        return round((position - minp) / self.resolution)

    # 给每个网格编号，得到每个网格的key
    def calc_index(self, node):
        # 从左到右增大，从下到上增大
        return node.y * self.x_width + node.x

    # 邻居节点是否超出范围
    def verify_node(self, node):
        # 根据网格坐标检查节点是否有效（不在障碍物上，不超出地图范围）
        if node.x < 0 or node.y < 0 or node.x >= self.x_width or node.y >= self.y_width:
            return False
        if self.obstacle_map[node.y, node.x]:  # 注意：numpy数组是先y后x
            return False
        return True

    # 计算路径, parent属性记录每个节点的父节点
    def calc_final_path(self, goal_node, closed_set):
        # 先存放终点坐标（真实坐标）
        rx = [self.calc_position(goal_node.x, self.min_x)]
        ry = [self.calc_position(goal_node.y, self.min_y)]
        # 获取终点的父节点索引
        parent_index = goal_node.parent_index
        # 起点的父节点==-1
        while parent_index != -1:
            n = closed_set[parent_index]  # 在入库中选择父节点
            rx.append(self.calc_position(n.x, self.min_x))  # 节点的x坐标
            ry.append(self.calc_position(n.y, self.min_y))  # 节点的y坐标
            parent_index = n.parent_index  # 节点的父节点索引

        return rx, ry


def main():
    print("Starting main program...")

    obstacle_map = np.load("../map/obstacle_map.npy").astype(np.bool_)
    # obstacle_map = np.logical_not(obstacle_map)

    sx, sy = 345, 95  # 起点
    gx, gy = 470, 475  # 终点
    grid_size = 1  # 网格大小
    robot_radius = 0.3  # 机器人半径，根据障碍物密度调整

    # 绘图
    if show_animation:
        # plt.plot(ox, oy, '.k')  # 障碍物黑色
        plt.imshow(obstacle_map, cmap='gray', origin='upper')
        plt.plot(sx, sy, 'og')  # 起点绿色
        plt.plot(gx, gy, 'xb')  # 终点蓝色
        plt.grid(True)
        plt.axis('equal')  # 坐标轴刻度间距等长
        plt.show()

    a_start = A_Start(obstacle_map, grid_size, robot_radius)
    rx, ry = a_start.planning(sx, sy, gx, gy)

    obstacle_map = np.logical_not(obstacle_map)
    if show_animation:
        plt.imshow(obstacle_map, cmap='gray', origin='upper')
        plt.plot(sx, sy, "og")  # 起点绿色
        plt.plot(gx, gy, "xb")  # 终点蓝色
        plt.plot(rx, ry, "-r")  # 路径红色
        plt.grid(True)
        plt.axis("equal")
        plt.show()


if __name__ == '__main__':
    main()
