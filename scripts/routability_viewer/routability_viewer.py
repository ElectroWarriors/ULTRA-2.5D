import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import os
import re
import concurrent.futures

# File paths
log_path = "/path/to/Multi_Reticle/experiment/case_22c/case_msg/log_20251106-141617/"

indexs_plot = [0, 1, 2, 3, 4]

for index_plot in indexs_plot: 
    print(f'plot index: {index_plot}')
    grid_file_path = log_path + f'routability/grid_{index_plot}.txt'
    path_file_path = log_path + f'routability/path_{index_plot}.txt'
    hotspot_file_path = log_path + f'routability/hotspot_{index_plot}.txt'
    output_image_path = log_path + f'routability/grid_paths_visualization_{index_plot}.png'

    # Read grid data
    grids = []
    dem_x_ratios = []
    dem_y_ratios = []

    with open(grid_file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) != 8:
                continue
            xc, yc, w, h, cap_x, cap_y, dem_x, dem_y = map(float, tokens)
            ratio_x = dem_x # / cap_x if cap_x > 0 else -1
            ratio_y = dem_y # / cap_y if cap_y > 0 else -1
            grids.append((xc, yc, w, h, cap_x, cap_y, ratio_x, ratio_y))
            if ratio_x >= 0:
                dem_x_ratios.append(ratio_x)
            if ratio_y >= 0:
                dem_y_ratios.append(ratio_y)

    # Normalize for color mapping
    norm_x = mcolors.Normalize(vmin=min(dem_x_ratios), vmax=max(dem_x_ratios))
    norm_y = mcolors.Normalize(vmin=min(dem_y_ratios), vmax=max(dem_y_ratios))
    cmap = cm.get_cmap('Reds')

    # Read path data
    paths = []
    with open(path_file_path, 'r') as f:
        for line in f:
            if line.startswith("## Path"):
                coords = line.split(":")[1].strip()
                points = []
                for token in coords.split(")"):
                    if "(" in token:
                        x, y = map(float, token.strip("()").split(","))
                        points.append((x, y))
                paths.append(points)

    with open(hotspot_file_path, 'r') as f:
        lines = f.readlines()
    def extract_rects(line):
        return [tuple(map(float, s.split(','))) for s in re.findall(r'\(([^)]+)\)', line)]
    horizontal_rects = extract_rects(lines[0])
    vertical_rects = extract_rects(lines[1])

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 20))

    # Draw grids in X direction (top)
    for xc, yc, w, h, cap_x, _, ratio_x, _ in grids:
        color = 'black' if cap_x == 0 else cmap(norm_x(ratio_x))
        rect = patches.Rectangle((xc - w/2, yc - h/2), w, h,
                                linewidth=0.3, edgecolor='gray', facecolor=color)
        ax1.add_patch(rect)

    # Draw grids in Y direction (bottom)
    for xc, yc, w, h, _, cap_y, _, ratio_y in grids:
        color = 'black' if cap_y == 0 else cmap(norm_y(ratio_y))
        rect = patches.Rectangle((xc - w/2, yc - h/2), w, h,
                                linewidth=0.3, edgecolor='gray', facecolor=color)
        ax2.add_patch(rect)

    # Draw paths on both subplots
    for path in paths:
        xs, ys = zip(*path)
        ax1.plot(xs, ys, color='green', linewidth=0.05)
        ax2.plot(xs, ys, color='green', linewidth=0.05)

    # draw horizontal hotspots
    for x, y, w, h in horizontal_rects:
        rect = patches.Rectangle((x - w/2, y - h/2), w, h,
                                linewidth=0.8, edgecolor='blue', facecolor='none')
        ax1.add_patch(rect)

    # draw vertical hotspots
    for x, y, w, h in vertical_rects:
        rect = patches.Rectangle((x - w/2, y - h/2), w, h,
                                linewidth=0.8, edgecolor='blue', facecolor='none')
        ax2.add_patch(rect)


    # Final touches
    for ax in [ax1, ax2]:
        ax.set_aspect('equal')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(False)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 200)

    ax1.set_title("Grid Demand/Capacity in X Direction")
    ax2.set_title("Grid Demand/Capacity in Y Direction")

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    # plt.show()
