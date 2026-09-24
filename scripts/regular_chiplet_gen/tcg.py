# import numpy as np
# import json


# Nx_chiplet = 15
# Ny_chiplet = 15
# x_range = [0, 12000]
# y_range = [0, 12000]

# x_edges = np.linspace(x_range[0], x_range[1], Nx_chiplet + 1)
# y_edges = np.linspace(y_range[0], y_range[1], Ny_chiplet + 1)
# x_size_percentage = 0.8
# y_size_percentage = 0.8

# # print(x_edges)
# # print(y_edges)
# json_discription = {"interposer": {"sizeInterposer": [38762.04, 32641.02], "numTSV": [155, 130], "sizeTSV": [40, 40], "minPitch": 250}, "chiplet": []}
# for j in range(len(y_edges)-1):
#     for i in range(len(x_edges)-1):
#         xc = (x_edges[i] + x_edges[i+1]) / 2
#         yc = (y_edges[j] + y_edges[j+1]) / 2
#         w  = (x_edges[i+1] - x_edges[i]) * x_size_percentage
#         h  = (y_edges[j+1] - y_edges[j]) * y_size_percentage
#         json_discription["chiplet"].append(f'{{"id": {j*Ny_chiplet+i}, "sizeChiplet": [{w}, {h}], "loc": [{xc}, {yc}], "numPin": [15, 15], "sizePin": [2, 2], "netIds": [2, 3]}}')

# with open("/path/to/pyStorm/scripts/regular_chiplet_gen/out.json", "w") as f:
#     json.dump(json_discription, f, indent=4)

import numpy as np
import json

Nx_chiplet = 15
Ny_chiplet = 15
x_range = [0, 60000]
y_range = [0, 60000]

x_edges = np.linspace(x_range[0], x_range[1], Nx_chiplet + 1)
y_edges = np.linspace(y_range[0], y_range[1], Ny_chiplet + 1)
x_size_percentage = 0.8
y_size_percentage = 0.8

interposer_info = {
    "sizeInterposer": [60000, 60000],
    "numTSV": [155, 130],
    "sizeTSV": [40, 40],
    "minPitch": 250
}

dtc_info = {"size": [10, 10], "value": 10}

chiplets = []
for j in range(len(y_edges) - 1):
    for i in range(len(x_edges) - 1):
        xc = (x_edges[i] + x_edges[i + 1]) / 2
        yc = (y_edges[j] + y_edges[j + 1]) / 2
        w = (x_edges[i + 1] - x_edges[i]) * x_size_percentage
        h = (y_edges[j + 1] - y_edges[j]) * y_size_percentage
        chiplet_info = {
            "id": j * Ny_chiplet + i,
            "sizeChiplet": [w, h],
            "loc": [xc, yc],
            "numPin": [80, 80],
            "sizePin": [2, 2],
            "netIds": [2, 3]
        }
        chiplets.append(chiplet_info)

connection_matrix = []
weight_matrix = []
for j in range(Ny_chiplet):
    for i in range(Nx_chiplet):
        a = j * Nx_chiplet + i
        if i + 1 < Nx_chiplet:
            b = j * Nx_chiplet + (i + 1)
            critical = 0.1 if (i == round(Nx_chiplet / 2)) else 9
            connection_matrix.append([a, b, 64])
            weight_matrix.append([a, b,  critical])
        if j + 1 < Ny_chiplet:
            b = (j + 1) * Nx_chiplet + i
            critical = 0.1 if (j == round(Ny_chiplet / 2)) else 9
            connection_matrix.append([a, b, 64])
            weight_matrix.append([a, b,  critical])


# 手动写入 JSON 文件，每个 chiplet 一行
with open("/path/to/Multi_Reticle/scripts/regular_chiplet_gen/out.json", "w") as f:
    # interposer 一行写入
    f.write('{\n')
    f.write('    "interposer": ' + json.dumps(interposer_info, separators=(',', ': ')) + ',\n')
    f.write('    "dtc": ' + json.dumps(dtc_info, separators=(',', ': ')) + ',\n')
    
    # chiplet 每行一个
    f.write('    "chiplets": [\n')
    for idx, chiplet in enumerate(chiplets):
        line = '        ' + json.dumps(chiplet, separators=(',', ': '))
        if idx < len(chiplets) - 1:
            line += ','
        f.write(line + '\n')
    f.write('    ],\n')

    f.write('    "use_sparse": 1,\n')

    f.write('    "connection_matrix": [\n')
    for idx, connection in enumerate(connection_matrix):
        line = '        ' + json.dumps(connection)
        if idx < len(connection_matrix) - 1:
            line += ','
        f.write(line + '\n')
    f.write('    ],\n')

    f.write('    "weight_matrix": [\n')
    for idx, connection in enumerate(weight_matrix):
        line = '        ' + json.dumps(connection)
        if idx < len(weight_matrix) - 1:
            line += ','
        f.write(line + '\n')
    f.write('    ]\n')

    f.write('}\n')
