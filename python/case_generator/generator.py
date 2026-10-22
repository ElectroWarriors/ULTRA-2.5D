import os
import time
import random
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json

import python.utils.utils as utils


class Chiplets(object):
    def __init__(self, work_dir, case_dir, enplot=False):
        # save work dir
        self.work_dir = work_dir
        self.case_dir = case_dir

        self.save_dir = work_dir + case_dir + "json_chiplets/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)

        # TSV model dir and data
        self.tsv_model = work_dir+case_dir+"mdl_v2_point.txt"
        self.tsv_model_data = np.loadtxt(self.tsv_model)
        # time
        self.start = time.time()
        # save input json discription
        self.json_discription_file = work_dir+case_dir+"place_init/case_discription.json"
        self.json_discription = utils.read_json(self.json_discription_file)
        self.json_interposer = self.json_discription["interposer"]
        self.json_chiplets = self.json_discription["chiplets"]
        self.json_dtc = self.json_discription["dtc"]
        # output json(interposer) init
        self.json_case_interposer = {"sizeInterposer": self.json_interposer["sizeInterposer"], "TSVs": []}
        # output json(chiplets) init
        self.json_case_chiplets = {"chiplets": [], "pins": [], "nets": []}
        self.json_case_chiplets["nets"].append({"id": 0, "name": "net0", "isPower": 1, "cells_id": [], "pins_id":  []})
        self.json_case_chiplets["nets"].append({"id": 1, "name": "net1", "isPower": 1, "cells_id": [], "pins_id":  []})

        # check interposer
        sizeInterposer = self.json_interposer["sizeInterposer"]
        numTSV = self.json_interposer["numTSV"]
        sizeTSV = self.json_interposer["sizeTSV"]
        if (all(x != 0 for x in sizeInterposer) and all(x != 0 for x in numTSV) and all(x != 0 for x in sizeTSV)):
            TSVDistance = [a / b for a, b in zip(sizeInterposer, numTSV)]
        else:
            raise ValueError("size/num = 0 or too many netIds.")
        if (not all(a > b for a, b in zip(TSVDistance, sizeTSV))):
            raise ValueError("Too small distance between pins.")
        
        # check chiplets
        for chipletIdx in range(len(self.json_chiplets)):
            chiplet_i = self.json_chiplets[chipletIdx]
            sizeChiplet = chiplet_i["sizeChiplet"]
            numPin = chiplet_i["numPin"]
            sizePin = chiplet_i["sizePin"]
            netIds = list(range(chiplet_i["netIds"][0], chiplet_i["netIds"][1]+1))

            if (all(x != 0 for x in sizeChiplet) and all(x != 0 for x in numPin) and 
                all(x != 0 for x in sizePin)  and numPin[0]*numPin[1] > len(netIds)):
                pinDistance = [a / b for a, b in zip(sizeChiplet, numPin)]
            else:
                raise ValueError("size/num = 0 or too many netIds.")
            if (not all(a > b for a, b in zip(pinDistance, sizePin))):
                raise ValueError("Too small distance between pins.")
        # some param about dtc
        self.random_dtc_map = True
        self.dtc_range = [100, 1000] # pF
        self.enplot = enplot

    
    def generate(self):
        self.interposer_generate()
        self.dtcmap_generate()
        self.chiplets_generate()
        if (self.enplot):
            self.plot()

    def interposer_generate(self):
        sizeInterposer = self.json_interposer["sizeInterposer"]
        numTSV = self.json_interposer["numTSV"]
        sizeTSV = self.json_interposer["sizeTSV"]
        TSVDistance = [a / b for a, b in zip(sizeInterposer, numTSV)]

        TSVIdx = 0
        self.all_tsv_loc = []
        for xIdx in range(numTSV[0]):
            for yIdx in range(numTSV[1]):
                # left-down loc
                pinXLoc = TSVDistance[0] * (1/2 + xIdx) - sizeTSV[0] / 2
                pinYLoc = TSVDistance[1] * (1/2 + yIdx) - sizeTSV[1] / 2
                self.all_tsv_loc.append((pinXLoc, pinYLoc))
                is_in = self.is_inChiplet(pinXLoc, pinYLoc, sizeTSV[0], sizeTSV[1])
                if (is_in == 0): # not in chiplet
                    Ilo, Ihi = random.uniform(1.0e-3, 2.0e-3), random.uniform(4e-3, 5e-3)
                    densityDTC = random.uniform(15, 25) / (TSVDistance[0] * TSVDistance[1])
                    inChiplet = 0
                elif (is_in == 1): # in chiplet but not close to center
                    Ilo, Ihi = random.uniform(1.0e-3, 2.0e-3), random.uniform(10e-3, 12e-3)
                    densityDTC = random.uniform(100, 110) / (TSVDistance[0] * TSVDistance[1])
                    inChiplet = 1
                elif (is_in == 2): # in chiplet and close to center
                    Ilo, Ihi = random.uniform(1.5e-3, 2.5e-3), random.uniform(12e-3, 15e-3)
                    densityDTC = random.uniform(50, 50) / (TSVDistance[0] * TSVDistance[1])
                    inChiplet = 1
                # Ilo, Ihi = min(I_rand), max(I_rand)
                # densityDTC = self.needNumDTC(Ilo, Ihi) / (TSVDistance[0] * TSVDistance[1]) # 个数密度！
                # print(TSVIdx, densityDTC)

                json_TSV_i = {"id": TSVIdx, "name": "TSV"+str(TSVIdx), "layer": 0, "capDensity": densityDTC, 
                              "loc": [int(pinXLoc), int(pinYLoc)], "size": [int(i) for i in sizeTSV], "inChiplet": inChiplet, 
                              "net": random.choice(["0_power", "1_power"])}
                self.json_case_interposer["TSVs"].append(json_TSV_i)
                TSVIdx += 1
        utils.write_json(self.save_dir+"interposer.json", self.json_case_interposer)
        print("interposer generate success, cost: ", time.time()-self.start, "s")

    def dtcmap_generate(self):
        if (self.random_dtc_map):
            numTSV = self.json_interposer["numTSV"]
            valueDTC = self.json_dtc["value"]
            self.numDTCmap = np.ceil(np.random.uniform(low=self.dtc_range[0], high=self.dtc_range[1], size=(numTSV[0], numTSV[1])) / float(valueDTC))
            np.savetxt(self.save_dir+"dtcmap.txt", self.numDTCmap, delimiter='\t', fmt='%d')
        print("DTC generate success, cost: ", time.time()-self.start, "s")


    def chiplets_generate(self):
        # generate nets
        conn = self.json_discription["connection_matrix"]
        is_sparse = self.json_discription["use_sparse"]
        matrix = utils.tomatrix(conn, is_sparse)
        
        all_nets_num = sum(matrix[i][j] for i in range(len(matrix)) for j in range(i, len(matrix[i])))
        for netIdx in range(all_nets_num):
            self.json_case_chiplets["nets"].append({"id": netIdx+2, "name": "net"+str(netIdx+2), "isPower": 0, "cells_id": [], "pins_id":  []})
        # deal with chiplets and it's pins
        pinIdx = 0
        kdtree = KDTree(self.all_tsv_loc)
        for chipletIdx in range(len(self.json_chiplets)):
            chiplet_i = self.json_chiplets[chipletIdx]
            sizeChiplet = chiplet_i["sizeChiplet"]
            numPin = chiplet_i["numPin"]
            sizePin = chiplet_i["sizePin"]
            netIds = list(range(chiplet_i["netIds"][0], chiplet_i["netIds"][1]+1))
            pin_netIds = utils.get_unique_random_integers(pinIdx, pinIdx+numPin[0]*numPin[1]-1, len(netIds))
            pinDistance = [a / b for a, b in zip(sizeChiplet, numPin)]

            locChiplet = []
            if ("loc" in chiplet_i):
                locChiplet = chiplet_i["loc"]
            json_chiplet_i = {"id": chiplet_i["id"], "name": "cell"+str(chiplet_i["id"]), 
                              "loc": locChiplet, "size": sizeChiplet, "numPin": numPin, 
                              "pins_id": [], "nets_id": [], "layer": 0}
            
            # relationship between pin and net is arbitary, which canbe ignored
            for xIdx in range(numPin[0]):
                for yIdx in range(numPin[1]):
                    json_chiplet_i["pins_id"].append(pinIdx)
                    pinXLoc = int(pinDistance[0] * (1/2 + xIdx) - sizePin[0] / 2)
                    pinYLoc = int(pinDistance[1] * (1/2 + yIdx) - sizePin[1] / 2)
                    distance, tsvIdx = kdtree.query((int(pinXLoc+chiplet_i["loc"][0]), int(pinYLoc+chiplet_i["loc"][1])))
                    capDensity = self.json_case_interposer["TSVs"][tsvIdx]["capDensity"] # 个数密度
                    json_pin_i = {"id": pinIdx, "name": "pin"+str(pinIdx), "ioLayer": 0, 
                                  "cell_id": chipletIdx, "net_id": -1, "offset": [pinXLoc, pinYLoc], "capDensity": capDensity, 
                                  "ioShape": {"xLo": pinXLoc, "yLo": pinYLoc, "xHi": pinXLoc+sizePin[0], "yHi": pinYLoc+sizePin[1]}}
                    self.json_case_chiplets["pins"].append(json_pin_i)
                    pinIdx += 1
            self.json_case_chiplets["chiplets"].append(json_chiplet_i)
        utils.write_json(self.save_dir+"chiplet.json", self.json_case_chiplets)
        print("chiplet generate success, total cost: ", time.time()-self.start, "s")


    def is_inChiplet(self, x, y, sx, sy):
        for c in self.json_chiplets:
            chiplet_l, chiplet_r = c["loc"][0], c["loc"][0] + c["sizeChiplet"][0]
            chiplet_b, chiplet_t = c["loc"][1], c["loc"][1] + c["sizeChiplet"][1]
            chiplet_cx, chiplet_cy = (chiplet_l + chiplet_r)/2, (chiplet_b + chiplet_t)/2
            tsv_cx, tsv_cy = x + sx / 2, y + sy / 2
            dist_x, dist_y = abs(chiplet_cx - tsv_cx), abs(chiplet_cy-tsv_cy)

            if (dist_x <= c["sizeChiplet"][0]/4 and dist_y <= c["sizeChiplet"][1]/4):
                return 2
            # case0: 0.999, case_ai: 0.99
            elif (dist_x <= 0.99*c["sizeChiplet"][0]/2 and dist_y <= 0.999*c["sizeChiplet"][1]/2):
                return 1
        return 0


    def plot(self):
        fig = plt.figure()  
        axes = fig.add_subplot(1, 1, 1)
        # interposer
        sizeInterposer = self.json_case_interposer["sizeInterposer"]
        square = plt.Rectangle(xy=(0, 0), width=sizeInterposer[0], height=sizeInterposer[1], alpha=0.1, edgecolor='black')
        axes.add_patch(square)
        for TSV_i in self.json_case_interposer["TSVs"]:
            loc = TSV_i["loc"]
            size = TSV_i["size"]
            square = plt.Rectangle(xy=(loc[0], loc[1]), width=size[0], height=size[1], alpha=0.3)
            axes.add_patch(square)
        print("plot: deal with TSV success, cost: ", time.time()-self.start, "s")
        # chiplets
        for chipletIdx in range(len(self.json_case_chiplets["chiplets"])):
            chiplet_i = self.json_case_chiplets["chiplets"][chipletIdx]
            loc = chiplet_i["loc"]
            size = chiplet_i["size"]
            square = plt.Rectangle(xy=(loc[0], loc[1]), width=size[0], height=size[1], alpha=0.5, edgecolor='black')
            axes.add_patch(square)
            for pinIdx in chiplet_i["pins_id"]:
                pin_i = self.json_case_chiplets["pins"][pinIdx]
                pin_loc = pin_i["offset"]
                pin_ioShape = pin_i["ioShape"]
                square = plt.Rectangle(xy=(loc[0]+pin_loc[0], loc[1]+pin_loc[1]), alpha=0.7, 
                                       width=pin_ioShape["xHi"]-pin_ioShape["xLo"], 
                                       height=pin_ioShape["yHi"]-pin_ioShape["yLo"])
                axes.add_patch(square)
            print(f"plot: deal with chiplet{chipletIdx} success, cost: ", time.time()-self.start, "s")
        print("plot: deal with chiplets success, cost: ", time.time()-self.start, "s")
        # nets
        # 创建一个映射，用于存储已遇到的 cell ID 组合及其对应的颜色
        cell_combinations_to_colors = {}
        n_colors = 0  # 颜色数量
        for net_i in self.json_case_chiplets["nets"]:
            cells_id_tuple = tuple(sorted(net_i["cells_id"]))  # 排序以确保 (0, 1) 和 (1, 0) 视为同一组合
            if cells_id_tuple not in cell_combinations_to_colors:
                cell_combinations_to_colors[cells_id_tuple] = n_colors
                n_colors += 1
        # 定义从红色到黑色的渐变颜色映射
        colors = [(1, 1, 0), (0, 1, 0)]  # 黄色到绿色
        custom_cmap = LinearSegmentedColormap.from_list('red_to_black', colors, N=n_colors)

        for net_i in self.json_case_chiplets["nets"]:
            if (net_i["id"] == 0 or net_i["id"] == 1):
                continue
            net_i_cell0_loc = self.json_case_chiplets["chiplets"][net_i["cells_id"][0]]["loc"]
            net_i_cell1_loc = self.json_case_chiplets["chiplets"][net_i["cells_id"][1]]["loc"]
            net_i_pin0_loc  = self.json_case_chiplets["pins"][net_i["pins_id"][0]]["offset"]
            net_i_pin1_loc  = self.json_case_chiplets["pins"][net_i["pins_id"][1]]["offset"]
            net_i_pin0_ioShape = self.json_case_chiplets["pins"][net_i["pins_id"][0]]["ioShape"]
            net_i_pin1_ioShape = self.json_case_chiplets["pins"][net_i["pins_id"][1]]["ioShape"]
            net_i_pin0_size = [net_i_pin0_ioShape["xHi"]-net_i_pin0_ioShape["xLo"], net_i_pin0_ioShape["yHi"]-net_i_pin0_ioShape["yLo"]]
            net_i_pin1_size = [net_i_pin1_ioShape["xHi"]-net_i_pin1_ioShape["xLo"], net_i_pin1_ioShape["yHi"]-net_i_pin1_ioShape["yLo"]]
            pin0_global_loc = [net_i_cell0_loc[0]+net_i_pin0_loc[0]+net_i_pin0_size[0]/2, net_i_cell0_loc[1]+net_i_pin0_loc[1]+net_i_pin0_size[1]/2]
            pin1_global_loc = [net_i_cell1_loc[0]+net_i_pin1_loc[0]+net_i_pin1_size[0]/2, net_i_cell1_loc[1]+net_i_pin1_loc[1]+net_i_pin1_size[1]/2]

            # 将 cells_id 转换为不可变的元组作为字典键
            cells_id_tuple = tuple(sorted(net_i["cells_id"]))  # 排序以确保 (0, 1) 和 (1, 0) 视为同一组合
            color_index = cell_combinations_to_colors[cells_id_tuple]
            color = custom_cmap(color_index / (n_colors - 1)) if n_colors > 1 else colors[0]
        
            # line = plt.plot([pin0_global_loc[0], pin1_global_loc[0]], [pin0_global_loc[1], pin1_global_loc[1]], color=color, linewidth=0.005, alpha=0.3)
            square = plt.Rectangle(xy=(pin0_global_loc[0], pin0_global_loc[1]), edgecolor="red", linewidth=0.008, facecolor="none", alpha=0.008, 
                                   width=pin1_global_loc[0]-pin0_global_loc[0], height=pin1_global_loc[1]-pin0_global_loc[1])
        plt.axis('equal') 
        print("save figure...")
        plt.savefig(self.save_dir+"plot.pdf", format='pdf', bbox_inches='tight')
        print("plot: save figure success, cost: ", time.time()-self.start, "s")
        # print("show figure...")
        # plt.show()
        # print("plot: show figure success, cost: ", time.time()-self.start, "s")


    def tsv_model_interpolation(self, x0_new, x1_new, showflag):
        x0 = self.tsv_model_data[:, 0]
        x1 = self.tsv_model_data[:, 1]
        x2 = self.tsv_model_data[:, 2]
        y  = self.tsv_model_data[:, 3]

        # 生成插值点
        # 固定 x0 和 x1，插值 x2 关于 y 的曲线
        x2_new = np.linspace(x2.min(), x2.max(), 100)  # 生成 x2 的插值点
        X_new = np.array([np.full_like(x2_new, x0_new), np.full_like(x2_new, x1_new), x2_new]).T

        # 使用 griddata 进行插值
        y_new = griddata((x0, x1, x2), y, X_new, method='nearest')

        if (showflag):
            # 可视化结果
            plt.plot(x2_new, y_new, 'r.-', label="插值曲线")
            plt.scatter(x2, y, label="original data")
            plt.xlabel("x2")
            plt.ylabel("y")
            plt.title(f"x2 关于 y 的插值曲线 (x0={x0_new}, x1={x1_new})")
            plt.legend()
            plt.grid()
            plt.show()
        return x2_new, y_new
    
    def needNumDTC(self, x0_new, x1_new):
        # DTC_size = self.json_dtc["size"]
        DTC_value = self.json_dtc["value"] * (1e-12)
        xs, ys = self.tsv_model_interpolation(x0_new, x1_new, showflag=False)
        for idx in range(len(ys)):
            if (ys[idx] < 0.05):
                print(xs[idx], ys[idx])
                return int(xs[idx]/DTC_value)
        return int(xs[-1]/DTC_value)