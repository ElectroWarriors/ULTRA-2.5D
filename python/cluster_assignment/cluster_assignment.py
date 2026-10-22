import os
import random

import python.cluster_assignment.cca as cca
import python.utils.utils as utils


class ClusterAssignment(object):
    def __init__(self, work_dir, case_dir):
        self.work_dir = work_dir
        self.case_dir = case_dir
        self.io_dir = work_dir+case_dir+"json_chiplets/"

        self.json_file = work_dir+case_dir+"place_init/case_discription.json"
        self.json_discription = utils.read_json(self.json_file)
        self.chiplets_file = self.io_dir+"chiplet.json"
        self.chiplets = utils.read_json(self.chiplets_file)

        conn = self.json_discription["connection_matrix"]
        is_sparse = self.json_discription["use_sparse"]
        self.cluster_con_matrix = utils.tomatrix(conn, is_sparse)
        
        k = len(self.cluster_con_matrix)
        # 编号
        self.cluster_ids_matrix = [[((i*(k-1)+j-i*(i+1)//2) if i < j else ((j*(k-1) + i-j*(j+1)//2) if i > j else 0)) for j in range(k)] for i in range(k)]
        self.cluster_con_vector = [self.cluster_con_matrix[i][j] for i in range(len(self.cluster_con_matrix)) for j in range(i+1, len(self.cluster_con_matrix[i]))]

    def get_center_loc(self, chipletIdx):
        chiplet_i = self.json_discription["chiplets"][chipletIdx]
        chiplet_i_loc = chiplet_i["loc"]
        chiplet_i_size = chiplet_i["sizeChiplet"]
        return [chiplet_i_loc[idx]+chiplet_i_size[idx]/2 for idx in range(2)]
    
    def get_overlap_xy(self, chipletIdx_i, chipletIdx_j):
        chiplet_i = self.json_discription["chiplets"][chipletIdx_i]
        chiplet_i_loc = chiplet_i["loc"]
        chiplet_i_size = chiplet_i["sizeChiplet"]
        chiplet_j = self.json_discription["chiplets"][chipletIdx_j]
        chiplet_j_loc = chiplet_j["loc"]
        chiplet_j_size = chiplet_j["sizeChiplet"]
        xx = max(chiplet_i_loc[0] + chiplet_i_size[0], chiplet_j_loc[0] + chiplet_j_size[0]) - \
             min(chiplet_i_loc[0], chiplet_j_loc[0]) - (chiplet_i_size[0] + chiplet_j_size[0])
        yy = max(chiplet_i_loc[1] + chiplet_i_size[1], chiplet_j_loc[1] + chiplet_j_size[1]) - \
             min(chiplet_i_loc[1], chiplet_j_loc[1]) - (chiplet_i_size[1] + chiplet_j_size[1])
        return xx, yy
    
    def calculate_mu(self, chipletIdx_i, chipletIdx_j):
        # center_i = self.get_center_loc(chipletIdx_i)
        # center_j = self.get_center_loc(chipletIdx_j)
        chiplet_i = self.json_discription["chiplets"][chipletIdx_i]
        chiplet_i_loc = chiplet_i["loc"]
        chiplet_i_size = chiplet_i["sizeChiplet"]
        chiplet_j = self.json_discription["chiplets"][chipletIdx_j]
        chiplet_j_loc = chiplet_j["loc"]
        chiplet_j_size = chiplet_j["sizeChiplet"]
        xx = sorted([chiplet_i_loc[0], chiplet_i_loc[0]+chiplet_i_size[0], chiplet_j_loc[0], chiplet_j_loc[0]+chiplet_j_size[0]])
        yy = sorted([chiplet_i_loc[1], chiplet_i_loc[1]+chiplet_i_size[1], chiplet_j_loc[1], chiplet_j_loc[1]+chiplet_j_size[1]])
        print(xx, yy)
        return [(xx[1] + xx[2]) / 2, (yy[1] + yy[2]) / 2]
    
    def calculate_params(self, chipletIdx_i):
        params = []
        chiplet_i_numPin = self.json_discription["chiplets"][chipletIdx_i]["numPin"]
        chiplet_i_loc = self.json_discription["chiplets"][chipletIdx_i]["loc"]
        chiplet_i_size = self.json_discription["chiplets"][chipletIdx_i]["sizeChiplet"]
        chiplet_i_conn = self.cluster_con_matrix[chipletIdx_i]
        chiplet_i_conn_nzIdx = [idx for idx, val in enumerate(chiplet_i_conn) if val != 0]

        # print(chiplet_i_conn, self.cluster_ids_matrix[chipletIdx_i])
        cluster_ids = [self.cluster_ids_matrix[chipletIdx_i][i] for i, value in enumerate(chiplet_i_conn) if value != 0]

        if (len(chiplet_i_conn_nzIdx) == 0):
            raise ValueError("isolate chiplet.")
        # elif (len(chiplet_i_conn_nzIdx) == 1):
        #     # isPower = 1
        #     n_pins = self.json_discription["chiplets"][chiplet_i_conn_nzIdx[0]]["numPin"]
        #     n_nets = n_pins[0]*n_pins[1] - chiplet_i_conn[chiplet_i_conn_nzIdx[0]]
        #     mu = self.get_center_loc(chipletIdx_i)
        #     sigma = [n_nets/2, n_nets/2]
        #     prefer_sigma = [0.5, 0.5]
        #     params.append(mu + sigma + prefer_sigma + [n_nets])
        # else:
            # grid_size = prefer_sigma = self.json_discription["chiplets"][chipletIdx_i]["numPin"]
            # isPower = 0
        udlr_direction_id = [[], [], [], []]
        for chipletIdx_j in chiplet_i_conn_nzIdx:
            n_nets = chiplet_i_conn[chipletIdx_j]
            chiplet_j_loc = self.json_discription["chiplets"][chipletIdx_j]["loc"]
            chiplet_j_size = self.json_discription["chiplets"][chipletIdx_j]["sizeChiplet"]
            if (n_nets == 0):
                continue
            mu = self.calculate_mu(chipletIdx_i, chipletIdx_j)
            mu = [(mu[i]-chiplet_i_loc[i])*(chiplet_i_numPin[i]/chiplet_i_size[i]) for i in range(2)]
            ovl_x, ovl_y = self.get_overlap_xy(chipletIdx_i, chipletIdx_j)
            if (ovl_x >= 0 and ovl_y <= 0):
                prefer_sigma = [1/8, 7/8]
                mu[0] = chiplet_i_numPin[0] if chiplet_i_loc[0] <= chiplet_j_loc[0] else 0
                direction = 3 if chiplet_i_loc[0] <= chiplet_j_loc[0] else 2
            elif (ovl_x <= 0 and ovl_y >= 0):
                prefer_sigma = [7/8, 1/8]
                mu[1] = chiplet_i_numPin[1] if chiplet_i_loc[1] <= chiplet_j_loc[1] else 0
                direction = 0 if chiplet_i_loc[1] <= chiplet_j_loc[1] else 1
            elif (ovl_x >= 0 or ovl_y >= 0):
                if (chiplet_i_loc[0] < chiplet_j_loc[0] and chiplet_i_loc[1] < chiplet_j_loc[1]): # up
                    prefer_sigma = [7/8, 1/8]
                    mu[1] = chiplet_i_numPin[1]
                    direction = 0
                elif (chiplet_i_loc[0] < chiplet_j_loc[0] and chiplet_i_loc[1] > chiplet_j_loc[1]): # down
                    prefer_sigma = [7/8, 1/8]
                    mu[1] = 0
                    direction = 1
                elif (chiplet_i_loc[0] > chiplet_j_loc[0] and chiplet_i_loc[1] > chiplet_j_loc[1]): # left
                    prefer_sigma = [1/8, 7/8]
                    mu[0] = 0
                    direction = 2
                elif (chiplet_i_loc[0] > chiplet_j_loc[0] and chiplet_i_loc[1] < chiplet_j_loc[1]): # left
                    prefer_sigma = [1/8, 7/8]
                    mu[0] = 0
                    direction = 2
            udlr_direction_id[direction].append(self.cluster_ids_matrix[chipletIdx_i][chipletIdx_j])

            A = n_nets
            n_nets_aver = sum(chiplet_i_conn)/len(chiplet_i_conn_nzIdx)
            sigma = [n_nets_aver/30, n_nets_aver/30]
            params.append([A] + mu + sigma + prefer_sigma + [2 * n_nets])
            # pa = cca.PinAssignment(grid_size=grid_size, target_counts=target_counts, params=params)
            # pa.run()
            # 补充以下Power和Signal的预分配（考虑SI，PI因素），然后修改以下cca关于num(pin_type)的统计方法（根据分配的Signal）
        return params, cluster_ids, udlr_direction_id
    
    def assign_chiplet_cluster(self):
        n_chiplet = len(self.cluster_con_matrix)
        for chipletIdx in range(n_chiplet):
            pins_cluster_rec = [0 for i in range(1+n_chiplet*(n_chiplet+1)//2)]
            chiplet_i, index = next(((item, idx) for idx, item in enumerate(self.chiplets["chiplets"]) if item["id"] == chipletIdx), None)
            chiplet_des_i = next((item for item in self.json_discription["chiplets"] if item["id"] == chipletIdx), None)
            if (not (chipletIdx == index == chiplet_i["id"] == chiplet_des_i["id"])):
                raise ValueError("...")

            print(f'------------- chiplet{index} -------------')
            if (len(chiplet_i["pins_id"]) != chiplet_des_i["numPin"][0]*chiplet_des_i["numPin"][1]):
                raise ValueError("...")
            grid_size = chiplet_des_i["numPin"]
            params, cluster_ids, udlr_direction_id = self.calculate_params(index)
            # print(params, cluster_ids)
            pa = cca.PinAssignment(grid_size=grid_size, cluster_ids=cluster_ids, params=params, 
                                   udlr_direction_id=udlr_direction_id, 
                                   save_dir=self.io_dir+"log_cluster_reassignment/")
            types = pa.run(chiplet_id=chipletIdx)
            m, n = types.shape
            pins_id = iter(sorted(chiplet_i["pins_id"]))
            pg_idx = 0
            for i in range(n):
                for j in range(m):
                    idx = types[j, i]
                    pin_id = next(pins_id)
                    if (idx != -1 and pins_cluster_rec[idx-1] < self.cluster_con_vector[idx-1] and (i+j) % 2 != 0):
                        net_id = 2 + sum(self.cluster_con_vector[0:idx-1]) + pins_cluster_rec[idx-1]
                        self.chiplets["pins"][pin_id]["net_id_str"] = str(idx+1) + "_" + str(pins_cluster_rec[idx-1]) # TODO: delete
                        self.chiplets["pins"][pin_id]["typePower"] = int(idx) # means cluster type, net_id != 0/1
                        self.chiplets["pins"][pin_id]["net_id"] = net_id
                        self.chiplets["nets"][net_id]["pins_id"].append(pin_id)
                        self.chiplets["nets"][net_id]["cells_id"].append(index)
                        pins_cluster_rec[idx-1] += 1
                    else:
                        net_id = pg_idx # random.randint(0, 1)
                        pg_idx = 1 - pg_idx
                        if idx != -1:
                            self.chiplets["pins"][pin_id]["net_id_str"] = str(idx+1) + "_" + str(net_id)+"_power" # TODO: delete
                            self.chiplets["pins"][pin_id]["typePower"] = int(idx) # cluster type, means cluster vdd/gnd
                        else:
                            self.chiplets["pins"][pin_id]["net_id_str"] = str(net_id)+"_power" # TODO: delete
                            self.chiplets["pins"][pin_id]["typePower"] = 0 # 0, means global vdd/gnd
                        self.chiplets["pins"][pin_id]["net_id"] = net_id # net_id == 0/1 
                        self.chiplets["nets"][net_id]["pins_id"].append(pin_id)
                        self.chiplets["nets"][net_id]["cells_id"].append(index)
                        continue
            utils.write_json(self.chiplets_file, self.chiplets)
        






