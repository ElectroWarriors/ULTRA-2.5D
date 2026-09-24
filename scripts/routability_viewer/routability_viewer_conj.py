import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import os
import re

# File paths
log_path = "/path/to/Multi_Reticle/experiment/case_22c/case_msg/log_20251101-142003/"
indexs_plot = [0, 1, 2, 3, 4]

for index_plot in indexs_plot:
    print(f'plot index: {index_plot}')
    grid_file_path = log_path + f'routability/grid_{index_plot}.txt'
    path_file_path = log_path + f'routability/path_{index_plot}.txt'
    hotspot_file_path = log_path + f'routability/hotspot_{index_plot}.txt'
    output_image_path = log_path + f'routability/grid_paths_visualization_combined_{index_plot}.png'

    # === Read grid data ===
    grids = []
    ratios = []
    with open(grid_file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) != 8:
                continue
            xc, yc, w, h, cap_x, cap_y, dem_x, dem_y = map(float, tokens)

            # ratio = (dem_x/cap_x + dem_y/cap_y)
            ratio = 0.0
            if cap_x > 0:
                ratio += dem_x / cap_x
            if cap_y > 0:
                ratio += dem_y / cap_y

            grids.append((xc, yc, w, h, cap_x, cap_y, dem_x, dem_y, ratio))
            if (cap_x > 0 or cap_y > 0):
                ratios.append(ratio)

    if len(ratios) == 0:
        print(f"Warning: no valid ratios in grid_{index_plot}.txt")
        continue

    # === Normalize color map ===
    norm = mcolors.Normalize(vmin=min(ratios), vmax=max(ratios))
    cmap = cm.get_cmap('Reds')

    # === Read path data ===
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
                if points:
                    paths.append(points)

    # === Read hotspot data ===
    with open(hotspot_file_path, 'r') as f:
        lines = f.readlines()

    def extract_rects(line):
        return [tuple(map(float, s.split(','))) for s in re.findall(r'\(([^)]+)\)', line)]

    horizontal_rects = extract_rects(lines[0]) if len(lines) > 0 else []
    vertical_rects = extract_rects(lines[1]) if len(lines) > 1 else []

    # === Plot combined view ===
    fig, ax = plt.subplots(figsize=(12, 10))

    # 背景填充为浅灰色
    ax.set_facecolor('#BFBFBF')

    # Draw grids
    for xc, yc, w, h, cap_x, cap_y, dem_x, dem_y, ratio in grids:
        if cap_x == 0 and cap_y == 0:
            # 无容量区域：chiplet，用蓝色标示
            color = '#00B0F0'
        else:
            # 拥塞区域：使用 Reds colormap
            color = cmap(norm(ratio))
        rect = patches.Rectangle(
            (xc - w/2, yc - h/2), w, h,
            linewidth=0.3, edgecolor='gray', facecolor=color
        )
        ax.add_patch(rect)

    # Draw all paths（绿色）
    for path in paths:
        xs, ys = zip(*path)
        ax.plot(xs, ys, color='green', linewidth=0.05)

    # Draw hotspots（蓝色边框）
    for rects in [horizontal_rects, vertical_rects]:
        for x, y, w, h in rects:
            rect = patches.Rectangle(
                (x - w/2, y - h/2), w, h,
                linewidth=0.8, edgecolor='blue', facecolor='none'
            )
            ax.add_patch(rect)

    # === Final touches ===
    ax.set_aspect('equal')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(False)
    ax.set_xlim(0, 200)
    ax.set_ylim(0, 200)
    ax.set_title("Combined Grid Congestion (dem_x/cap_x + dem_y/cap_y)")

    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("(dem_x/cap_x + dem_y/cap_y)")

    plt.tight_layout()
    plt.savefig(output_image_path, dpi=300)
    plt.close()

    print(f"Saved: {output_image_path}")
