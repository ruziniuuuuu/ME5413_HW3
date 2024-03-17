# Reflections

This is a markdown file for the reflections of the project. It will be updated as the project progresses.

## About calculating the obstacle positions

We've noticed that it is very time-consuming to calculate the obstacle postions, as the following code shows:

```python
def calc_obstacle_map(self, ox, oy):

    self.min_x = round(min(ox))
    self.min_y = round(min(oy))
    self.max_x = round(max(ox))
    self.max_y = round(max(oy))
    print("min_x:", self.min_x)
    print("min_y:", self.min_y)
    print("max_x:", self.max_x)
    print("max_y:", self.max_y)

    self.x_width = round((self.max_x - self.min_x) / self.resolution)
    self.y_width = round((self.max_y - self.min_y) / self.resolution)
    print("x_width:", self.x_width)
    print("y_width:", self.y_width)

    # obstacle map generation
    self.obstacle_map = [[False for _ in range(self.y_width)]
                            for _ in range(self.x_width)]
    for ix in range(self.x_width):
        x = self.calc_grid_position(ix, self.min_x)
        for iy in range(self.y_width):
            y = self.calc_grid_position(iy, self.min_y)
            for iox, ioy in zip(ox, oy):
                d = math.hypot(iox - x, ioy - y)
                if d <= self.rr:
                    self.obstacle_map[ix][iy] = True
                    break
```

So we choose to store the calculated obstacle map in a `obsacle_map.pkl` file, and load it when we wanna re-create the Planner object. This is the modified code:

```python
        # obstacle map generation
        try:
            with open(self.obstacle_map_filename, "rb") as f:
                self.obstacle_map = pickle.load(f)
            print("Loaded obstacle map from file")
            return
        except FileNotFoundError:
            pass

        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break
        
        # Save the generated obstacle map to a file
        with open(self.obstacle_map_filename, "wb") as f:
            pickle.dump(self.obstacle_map, f)
        print("Saved obstacle map to file.")
```
