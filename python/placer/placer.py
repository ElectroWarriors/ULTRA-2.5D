import os
import time
import subprocess

import json
import pyomo.environ as env
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

import python.utils.utils as utils

class Placer(object):
    def __init__(self, work_dir, case_dir):
        self.work_dir = work_dir
        self.case_dir = case_dir

    def gd_placer(self):
        start = time.time()
        path_exe = "/path/to/Multi_Reticle/src/cu_placer/bin/cu_placer"
        subprocess.run([path_exe, self.work_dir, self.case_dir], cwd=self.work_dir)
        print("nlp_gplacer finished, cost:", time.time()-start)

    def milp_placer(self):
        start = time.time()

        json_chiplet    = utils.read_json(self.work_dir+self.case_dir+"json_chiplets/chiplet.json")
        json_interposer = utils.read_json(self.work_dir+self.case_dir+"json_chiplets/interposer.json")
        
        output_dir = self.work_dir+self.case_dir+"placer/milp_placer/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        output_file = output_dir + "milp_placer.lp"

        if (True):
            cpairs = {}
            for net in json_chiplet["nets"]:
                pins_id = net["pins_id"]
                if net["id"] in [0, 1] or len(pins_id) < 2:
                    continue
                chiplet1_id = json_chiplet["pins"][pins_id[0]]["cell_id"]
                chiplet2_id = json_chiplet["pins"][pins_id[1]]["cell_id"]
                chiplet1_size   = json_chiplet["chiplets"][chiplet1_id]["size"].copy()
                chiplet2_size   = json_chiplet["chiplets"][chiplet2_id]["size"].copy()
                chiplet1_offset = json_chiplet["pins"][pins_id[0]]["offset"].copy()
                chiplet2_offset = json_chiplet["pins"][pins_id[1]]["offset"].copy()
                chiplet1_offset[0], chiplet1_offset[1] = chiplet1_offset[0] - chiplet1_size[0]/2, chiplet1_offset[1] - chiplet1_size[1]/2
                chiplet2_offset[0], chiplet2_offset[1] = chiplet2_offset[0] - chiplet2_size[0]/2, chiplet2_offset[1] - chiplet2_size[1]/2
                
                s_chiplet1_id, s_chiplet2_id = sorted([chiplet1_id, chiplet2_id])
                pair_key = (s_chiplet1_id, s_chiplet2_id)

                if pair_key not in cpairs:
                    cpairs[pair_key] = { "num_net": 0, "chiplet1_offset": [0, 0], "chiplet2_offset": [0, 0] }

                cpairs[pair_key]["num_net"] += 1

            for pair_key in cpairs:
                num_net = cpairs[pair_key]["num_net"]
                cpairs[pair_key]["chiplet1_offset"][0] /= num_net
                cpairs[pair_key]["chiplet1_offset"][1] /= num_net
                cpairs[pair_key]["chiplet2_offset"][0] /= num_net
                cpairs[pair_key]["chiplet2_offset"][1] /= num_net

            model = env.ConcreteModel()
            
            chiplets = list(set([chiplet_id for pair in cpairs.keys() for chiplet_id in pair]))
            model.C = env.Set(initialize=chiplets)
            
            connections = list(cpairs.keys())
            model.A = env.Set(initialize=connections)

            chiplet_widths = {}
            chiplet_heights = {}
            for chiplet in model.C:
                chiplet_widths[chiplet]  = json_chiplet["chiplets"][chiplet]["size"][0]
                chiplet_heights[chiplet] = json_chiplet["chiplets"][chiplet]["size"][1]
            
            model.w = env.Param(model.C, initialize=chiplet_widths)
            model.h = env.Param(model.C, initialize=chiplet_heights)
            
            W = json_interposer["sizeInterposer"][0]
            H = json_interposer["sizeInterposer"][1]
            
            BIG_M = W + H + 1000
            
            connection_weights = {}
            for (i, j) in model.A:
                connection_weights[(i, j)] = cpairs[(i, j)]["num_net"]
            
            model.A_ij = env.Param(model.A, initialize=connection_weights)
            
            O_x = {}
            O_y = {}
            for (i, j) in model.A:
                O_x[(i, j)] = cpairs[(i, j)]["chiplet1_offset"][0]
                O_y[(i, j)] = cpairs[(i, j)]["chiplet1_offset"][1]
                O_x[(j, i)] = cpairs[(i, j)]["chiplet2_offset"][0]
                O_y[(j, i)] = cpairs[(i, j)]["chiplet2_offset"][1]
            
            model.O_x = env.Param(model.C, model.C, default=0, initialize=O_x)
            model.O_y = env.Param(model.C, model.C, default=0, initialize=O_y)
            
            model.x = env.Var(model.C, domain=env.NonNegativeReals, bounds=(0, W))
            model.y = env.Var(model.C, domain=env.NonNegativeReals, bounds=(0, H))
            
            model.u = env.Var(model.C, domain=env.Binary)
            model.v = env.Var(model.C, domain=env.Binary)
            
            model.diff_x_plus = env.Var(model.A, domain=env.NonNegativeReals)
            model.diff_x_minus = env.Var(model.A, domain=env.NonNegativeReals)
            model.diff_y_plus = env.Var(model.A, domain=env.NonNegativeReals)
            model.diff_y_minus = env.Var(model.A, domain=env.NonNegativeReals)
            
            model.delta = env.Var(model.C, model.C, [1, 2, 3, 4], domain=env.Binary)
            
            model.is_rotated = env.Var(model.C, domain=env.Binary)
            
            # 约束：is_rotated = 1 当且仅当芯片旋转了90度或270度 (即u != v)
            def rotation_constraint_1(m, i):
                return m.is_rotated[i] >= m.u[i] - m.v[i]
            
            def rotation_constraint_2(m, i):
                return m.is_rotated[i] >= m.v[i] - m.u[i]
            
            def rotation_constraint_3(m, i):
                return m.is_rotated[i] <= m.u[i] + m.v[i]
            
            def rotation_constraint_4(m, i):
                return m.is_rotated[i] <= 2 - m.u[i] - m.v[i]
            
            model.rotation_constraint_1 = env.Constraint(model.C, rule=rotation_constraint_1)
            model.rotation_constraint_2 = env.Constraint(model.C, rule=rotation_constraint_2)
            model.rotation_constraint_3 = env.Constraint(model.C, rule=rotation_constraint_3)
            model.rotation_constraint_4 = env.Constraint(model.C, rule=rotation_constraint_4)
            
            def rotated_width_rule(m, i):
                return m.w[i] * (1 - m.is_rotated[i]) + m.h[i] * m.is_rotated[i]

            def rotated_height_rule(m, i):
                return m.h[i] * (1 - m.is_rotated[i]) + m.w[i] * m.is_rotated[i]
            
            model.w_prime = env.Expression(model.C, rule=rotated_width_rule)
            model.h_prime = env.Expression(model.C, rule=rotated_height_rule)
            
            def rotated_clump_x_rule(m, i, j):
                return (m.x[i] + m.O_x[i, j] * (1 - m.u[i] - m.v[i]) 
                        - m.O_y[i, j] * (m.v[i] - m.u[i]))
            
            def rotated_clump_y_rule(m, i, j):
                return (m.y[i] + m.O_x[i, j] * (m.v[i] - m.u[i]) 
                        + m.O_y[i, j] * (1 - m.u[i] - m.v[i]))
            
            model.X_ij = env.Expression(model.C, model.C, rule=rotated_clump_x_rule)
            model.Y_ij = env.Expression(model.C, model.C, rule=rotated_clump_y_rule)
            
            def linearize_abs_x_constraint(m, i, j):
                return m.X_ij[i, j] - m.X_ij[j, i] == m.diff_x_plus[(i, j)] - m.diff_x_minus[(i, j)]
            
            def linearize_abs_y_constraint(m, i, j):
                return m.Y_ij[i, j] - m.Y_ij[j, i] == m.diff_y_plus[(i, j)] - m.diff_y_minus[(i, j)]
            
            model.linearize_x = env.Constraint(model.A, rule=linearize_abs_x_constraint)
            model.linearize_y = env.Constraint(model.A, rule=linearize_abs_y_constraint)
            
            def objective_rule(m):
                wirelength = 0
                for (i, j) in m.A:
                    wirelength += m.A_ij[(i, j)] * (
                        m.diff_x_plus[(i, j)] + m.diff_x_minus[(i, j)] + 
                        m.diff_y_plus[(i, j)] + m.diff_y_minus[(i, j)]
                    )
                return wirelength
            
            model.obj = env.Objective(rule=objective_rule, sense='minimize')
            
            def boundary_constraint_x_lower(m, i):
                return m.x[i] >= m.w_prime[i] / 2
            
            def boundary_constraint_x_upper(m, i):
                return m.x[i] <= W - m.w_prime[i] / 2
            
            def boundary_constraint_y_lower(m, i):
                return m.y[i] >= m.h_prime[i] / 2
            
            def boundary_constraint_y_upper(m, i):
                return m.y[i] <= H - m.h_prime[i] / 2
            
            model.boundary_x_lower = env.Constraint(model.C, rule=boundary_constraint_x_lower)
            model.boundary_x_upper = env.Constraint(model.C, rule=boundary_constraint_x_upper)
            model.boundary_y_lower = env.Constraint(model.C, rule=boundary_constraint_y_lower)
            model.boundary_y_upper = env.Constraint(model.C, rule=boundary_constraint_y_upper)
            
            min_spacing = 500.0

            def non_overlap_constraint_1(m, i, j):
                if i < j:
                    return m.x[i] + m.w_prime[i]/2 + min_spacing <= m.x[j] - m.w_prime[j]/2 + W * (1 - m.delta[i, j, 1])
                else:
                    return env.Constraint.Skip

            def non_overlap_constraint_2(m, i, j):
                if i < j:
                    return m.x[j] + m.w_prime[j]/2 + min_spacing <= m.x[i] - m.w_prime[i]/2 + W * (1 - m.delta[i, j, 2])
                else:
                    return env.Constraint.Skip

            def non_overlap_constraint_3(m, i, j):
                if i < j:
                    return m.y[i] + m.h_prime[i]/2 + min_spacing <= m.y[j] - m.h_prime[j]/2 + H * (1 - m.delta[i, j, 3])
                else:
                    return env.Constraint.Skip

            def non_overlap_constraint_4(m, i, j):
                if i < j:
                    return m.y[j] + m.h_prime[j]/2 + min_spacing <= m.y[i] - m.h_prime[i]/2 + H * (1 - m.delta[i, j, 4])
                else:
                    return env.Constraint.Skip

            def non_overlap_logic(m, i, j):
                if i < j:
                    return sum(m.delta[i, j, k] for k in [1, 2, 3, 4]) >= 1
                else:
                    return env.Constraint.Skip
            
            model.non_overlap_1 = env.Constraint(model.C, model.C, rule=non_overlap_constraint_1)
            model.non_overlap_2 = env.Constraint(model.C, model.C, rule=non_overlap_constraint_2)
            model.non_overlap_3 = env.Constraint(model.C, model.C, rule=non_overlap_constraint_3)
            model.non_overlap_4 = env.Constraint(model.C, model.C, rule=non_overlap_constraint_4)
            model.non_overlap_logic = env.Constraint(model.C, model.C, rule=non_overlap_logic)
            
            model.write(output_file, io_options={'symbolic_solver_labels': True})
            
            print(f"MILP模型已保存为: {output_file}")

            path_exe = "/path/to/Multi_Reticle/src/milp_placer/bin/milp_placer"
            subprocess.run([path_exe, self.work_dir, self.case_dir, "placer/milp_placer/milp_placer.lp"], cwd=self.work_dir)

        if (True):
            fig, ax = plt.subplots(1, 1, figsize=(12, 10))
            min_x, max_x = float('inf'), float('-inf')
            min_y, max_y = float('inf'), float('-inf')
            with open(output_file+".result", "r") as f:
                for line in f:
                    parts = line.strip().split()
                    x, y = float(parts[1]), float(parts[2])
                    if (int(parts[3]) == 0 or int(parts[3]) == 180):
                        w, h = json_chiplet["chiplets"][int(parts[0])]["size"]
                    else:
                        h, w = json_chiplet["chiplets"][int(parts[0])]["size"]

                    rect_min_x = x - w/2
                    rect_max_x = x + w/2
                    rect_min_y = y - h/2
                    rect_max_y = y + h/2
                    
                    min_x = min(min_x, rect_min_x)
                    max_x = max(max_x, rect_max_x)
                    min_y = min(min_y, rect_min_y)
                    max_y = max(max_y, rect_max_y)
                    rect = Rectangle((rect_min_x, rect_min_y), w, h, linewidth=2, edgecolor='black', facecolor="b", alpha=0.7)
                    ax.add_patch(rect)
            margin_x = (max_x - min_x) * 0.1
            margin_y = (max_y - min_y) * 0.1
            ax.set_xlim(min_x - margin_x, max_x + margin_x)
            ax.set_ylim(min_y - margin_y, max_y + margin_y)
            ax.set_aspect('equal')
            ax.set_title('Chiplet Layout')
            plt.tight_layout()
            plt.savefig(output_file+".png", dpi=300, bbox_inches='tight')

        print("lp_gplacer finished, cost:", time.time()-start)


        def rot90_multiple(js_case, cid, num_rotations):
            if num_rotations == 0:
                return js_case
            
            xl, yl = js_case["chiplets"][cid]["loc"]
            w, h = js_case["chiplets"][cid]["size"]
            xc, yc = xl + w / 2, yl + h / 2

            if num_rotations % 2 == 1:
                new_w, new_h = h, w
            else:  
                new_w, new_h = w, h

            for pin_id in js_case["chiplets"][cid]["pins_id"]:
                ox, oy = js_case["pins"][pin_id]["offset"]
                
                dx, dy = ox - w / 2, oy - h / 2
                
                for _ in range(num_rotations):
                    dx, dy = -dy, dx 
                
                js_case["pins"][pin_id]["offset"][0] = dx + new_w / 2
                js_case["pins"][pin_id]["offset"][1] = dy + new_h / 2

            js_case["chiplets"][cid]["size"] = [new_w, new_h]
            js_case["chiplets"][cid]["loc"] = [xc - new_w / 2, yc - new_h / 2]

            return js_case

        if (True):
            with open(output_file+".result", "r") as f:
                for line in f:
                    parts = line.strip().split()
                    cid = int(parts[0])
                    num_rot = int(int(parts[3]) / 90)
                    json_chiplet = rot90_multiple(json_chiplet, cid, num_rot)
                    
                    x, y = float(parts[1]), float(parts[2])
                    [w, h] = json_chiplet["chiplets"][cid]["size"]
                    json_chiplet["chiplets"][cid]["loc"] = [x-w/2, y-h/2]
            utils.write_json(self.work_dir+self.case_dir+"placer/milp_placer/chiplet_milp.json", json_chiplet)
                

