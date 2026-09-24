import os
import time
import numpy as np
import subprocess
import deap.base, deap.creator, deap.tools, deap.algorithms
import matplotlib.pyplot as plt
import pyomo.environ as env
from python.utils.utils import read_json, tomatrix

class Huristic_Partition(object):
    def __init__(self, work_dir, case_dir, reticle_file):
        # load path
        self.work_dir = work_dir
        self.case_dir = case_dir
        self.save_dir = work_dir+case_dir+"hur_partition/"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        # load reticles
        reticles = np.load(reticle_file) # .npz file
        self.Cr = reticles["Cr"] # connectivity matrix of reticles
        self.Gr = reticles["Gr"] # [x, y, w, h] of every reticles
        self.Ar = np.array([r[2]*r[3] for r in self.Gr])
        self.Nr = len(self.Ar)
        np.savetxt(self.save_dir+"reticle_msg.txt", self.Gr, fmt='%d')
        # load chiplets
        self.json_discription = read_json(work_dir+case_dir+"place_init/case_discription.json")
        self.json_chiplets = read_json(work_dir+case_dir+"json_chiplets/chiplet.json")

        num_chiplet = len(self.json_discription["chiplets"])

        conn = self.json_discription["connection_matrix"]
        weight = self.json_discription["weight_matrix"]
        is_sparse = self.json_discription["use_sparse"]
        self.Cc = np.array(tomatrix(conn, is_sparse, num_chiplet))
        self.Wc = np.array(tomatrix(weight, is_sparse, num_chiplet))
        self.Ac = np.array([c["sizeChiplet"][0]*c["sizeChiplet"][1] for c in self.json_discription["chiplets"]])
        self.Nc = len(self.Ac)

        # weight and option for MIQP
        self.w_wl_cross = 1
        self.w_wl_area_aver = 1
        self.w_ch_cross = 100
        self.w_area_slack = 1

        self.en_reticle_conn = True
        self.en_stitching_conn = True
        self.en_critical_net = True
        self.en_area_limit = True

        self.w_obj = [1, 1, 1] # weight of every objects in optimization algorithm
        self.w_viol = [1, 1] # weight of every violation function in optimization algorithm
        self.alpha = 1.5 # sum of chiplet area in the same reticle should not exceed alpha*Ar
        self.viol_func = np.vectorize(self._viol_exp, signature='()->()') # select exp violation function

        self.simiter = -1
        self.start = time.time()
        self.logger = []     
    


    def gen_MIQP(self):
        Nc, Nr = self.Nc, self.Nr
        Cc, Wc, Cr = self.Cc, self.Wc, self.Cr
        Ac, Ar = self.Ac, self.Ar
        alpha = self.alpha
        w_obj = self.w_obj
        w_viol = self.w_viol

        model = env.ConcreteModel()

        # Sets
        model.I = env.RangeSet(0, Nc-1)
        model.J = env.RangeSet(0, Nr)  # 0 = stitching, 1~Nr = reticles

        # Variables
        model.X = env.Var(model.I, model.J, domain=env.Binary)
        model.delta_max = env.Var(domain=env.NonNegativeReals)
        model.slack = env.Var(env.RangeSet(1, Nr), domain=env.NonNegativeReals)

        # One-hot constraint
        model.one_hot = env.Constraint(model.I, rule=lambda m, i: sum(m.X[i, j] for j in m.J) == 1)

        connected_chiplet_pairs = [ (i, j) for i in range(Nc) for j in range(i+1, Nc) if Cc[i, j] > 0]
        all_reticle_pairs = [ (r1, r2) for r1 in range(1, Nr+1) for r2 in range(r1+1, Nr+1)]

        # Hard constraint 1: stitching connection
        if (self.en_reticle_conn):
            model.stitching_block = env.ConstraintList()
            for i, j in connected_chiplet_pairs:
                model.stitching_block.add(model.X[i, 0] + model.X[j, 0] <= 1)

        # Hard constraint 2: critical net
        if (self.en_critical_net):
            model.critical_net = env.ConstraintList()
            for i, j in connected_chiplet_pairs:
                if Wc[i, j] != 9: continue
                for r1, r2 in all_reticle_pairs:
                    model.critical_net.add(model.X[i, r1] + model.X[j, r2] <= 1)
                    model.critical_net.add(model.X[i, r2] + model.X[j, r1] <= 1)
        
        # Hard constraint 3: reticle connectivity
        if (self.en_stitching_conn):
            model.reticle_conn = env.ConstraintList()
            for i, j in connected_chiplet_pairs:
                for r1, r2 in all_reticle_pairs:
                    if (Cr[r1-1, r2-1] != 0): continue
                    model.reticle_conn.add(model.X[i, r1] + model.X[j, r2] <= 1)
                    model.reticle_conn.add(model.X[i, r2] + model.X[j, r1] <= 1)

        # Hard constraint 4: area limit
        if (self.en_area_limit):
            def area_limit_rule(m, r):
                return sum(Ac[i] * m.X[i, r] for i in m.I) <= alpha * Ar[r-1] + m.slack[r]
            model.area_limit = env.Constraint(env.RangeSet(1, Nr), rule=area_limit_rule)

        # delta_max constraint
        # model.delta_max_con = ConstraintList()
        # for r in range(1, Nr+1):
        #     used_area = sum(Ac[i] * model.X[i, r] for i in range(Nc))
        #     delta = Ar[r-1] - used_area
        #     model.delta_max_con.add(model.delta_max >= delta)


        model.area_mean = env.Var(domain=env.NonNegativeReals)
        model.area_dev_max = env.Var(domain=env.NonNegativeReals)

        model.area_mean_def = env.Constraint(expr = model.area_mean == sum(sum(Ac[i] * model.X[i, r] for i in model.I) for r in env.RangeSet(1, Nr)) / Nr)

        model.area_dev_con = env.ConstraintList()
        for r in range(1, Nr+1):
            used_area = sum(Ac[i] * model.X[i, r] for i in model.I)
            model.area_dev_con.add(model.area_dev_max >= (used_area - model.area_mean) / Ar[r-1])
            model.area_dev_con.add(model.area_dev_max >= (model.area_mean - used_area) / Ar[r-1])


        # Objective
        def objective_rule(m):
            # wirelength（归一化）
            total_conn = sum(Cc[i,j] for i in range(Nc) for j in range(Nc))
            wl_cross = sum(
                Cc[i,j] * Wc[i,j] * m.X[i,r1] * m.X[j,r2] / 2
                for i in range(Nc) for j in range(Nc)
                for r1 in range(1,Nr+1) for r2 in range(1,Nr+1)
                if r1 != r2
            )
            wl_cross_percentage = wl_cross / total_conn if total_conn > 0 else 0

            # area_variance = model.area_variance / max(Ar)

            # stitching 使用比例
            stitch_count = sum(m.X[i,0] for i in range(Nc))
            chiplet_cross_percentage = stitch_count / Nc

            # deltaAmax（归一化）
            deltaAmax_percentage = m.delta_max / max(Ar)

            # slack 惩罚（不归一化，已指数放大）
            slack_penalty = sum(m.slack[r] for r in range(1, Nr+1))

            return (
                self.w_wl_cross * wl_cross_percentage +
                self.w_wl_area_aver * model.area_dev_max +
                self.w_ch_cross * chiplet_cross_percentage +
                self.w_area_slack * slack_penalty
            )  # w_obj[2] * deltaAmax_percentage +
        
        model.obj = env.Objective(rule=objective_rule, sense=env.minimize)

        # 保存模型为 LP 文件
        model.write(self.save_dir+"milp_model.lp", io_options={"symbolic_solver_labels": True})

    def set_option(self, en_reticle_conn, en_stitching_conn, en_critical_net, en_area_limit):
        self.en_reticle_conn = en_reticle_conn
        self.en_stitching_conn = en_stitching_conn
        self.en_critical_net = en_critical_net
        self.en_area_limit = en_area_limit

    def set_weight(self, w_wl_cross, w_wl_area_aver, w_ch_cross, w_area_slack):
        self.w_wl_cross = w_wl_cross
        self.w_wl_area_aver = w_wl_area_aver
        self.w_ch_cross = w_ch_cross
        self.w_area_slack = w_area_slack

    def call_MIQP_solver(self):
        path_exe = "/path/to/pyStorm/src/milp_solver/bin/milp_solver"
        subprocess.run([path_exe, self.work_dir, self.case_dir], cwd=self.work_dir)

    def run_MIQP(self):
        # self.gen_MIQP()
        self.gen_MIQP()
        self.call_MIQP_solver()
    
    # --------------------------------------- GA ---------------------------------------------------
    def objective_function(self, x):
        self.simiter += 1
        X = np.zeros((self.Nc, self.Nr+1))
        for idx, xi in enumerate(x):
            X[idx, xi] = 1
        
        Cr_calc = X.T @ (self.Cc * self.Wc) @ X

        # hard cons1: two connected net shound not both on stitching reigon
        if (Cr_calc[0, 0]): return [np.inf]
        
        # hard cons2: critial net
        for i in range(self.Nc):
            for j in range(i+1, self.Nc):
                if (self.Wc[i, j] == 9 and x[i]*x[j] != 0 and x[i] != x[j]):
                    return [np.inf]
        # hard cons3: A_max

        # obj1: deltaAmax
        Ac_plus = self.Ac.copy()
        chiplet_cross_idx = np.where(X[:, 0] == 1)[0]
        for idx in chiplet_cross_idx:
            tmp_row = self.Cc[idx, :]
            Ac_plus += self.Ac[idx] * tmp_row / tmp_row.sum()
        Ac_in_reticle = Ac_plus @ X[:, 1:]
        deltaA = self.Ar - Ac_in_reticle
        deltaAmax_percentage = np.max(deltaA / self.Ar) # value

        # obj2: wirelength
        Cr_calc_inter = Cr_calc - np.diag(np.diag(Cr_calc))
        Cr_calc_intra = np.diag(Cr_calc) / 2
        wl_cross_percentage = np.sum(Cr_calc_inter[1:, 1:]) / np.sum(self.Cc) # value

        # obj3: number of cross reticle chiplets
        chiplet_cross_percentage = np.sum(X[:, 0] == 1) / self.Nc
        
        # cons1: disobey reticle connectivity
        rows, cols = np.triu_indices_from(self.Cr, k=1)
        values = Cr_calc[rows+1, cols+1][self.Cr[rows, cols] == 0]
        if np.any(values != 0): return [np.inf]
        viol_recicle_conn = values / (np.max(self.Cc)) if values.size else np.array([0])
        viol_recicle_conn = self.viol_func(viol_recicle_conn) # np array

        # cons2: area limit
        viol_area_limit = self.viol_func(2*(self.alpha-Ac_in_reticle/self.Ar)) # np array

        obj_val = np.array([deltaAmax_percentage, wl_cross_percentage, chiplet_cross_percentage]).reshape(-1, 1)
        viol_val = np.array([np.sum(viol_recicle_conn), np.sum(viol_area_limit)]).reshape(-1, 1)

        obj_viol = self.w_obj @ obj_val + self.w_viol @ viol_val
        
        logger_i = [time.time()-self.start, self.simiter] + x + obj_val.flatten().tolist() + viol_val.flatten().tolist() + obj_viol.tolist()
        self.logger.append(logger_i)
        print(logger_i)

        return obj_viol


    def _viol_exp(self, x):
        return 0 if x <= 0 else np.exp(x) - 1

    
    def run_GA(self):
        deap.creator.create("FitnessMin", deap.base.Fitness, weights=(-1.0,))
        deap.creator.create("Individual", list, fitness=deap.creator.FitnessMin)

        toolbox = deap.base.Toolbox()
        toolbox.register("attr_int", np.random.randint, 0, self.Nr+1)  # 离散变量：0~N-1 的整数
        toolbox.register("individual", deap.tools.initRepeat, deap.creator.Individual, toolbox.attr_int, self.Nc)
        toolbox.register("population", deap.tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.objective_function)
        toolbox.register("mate", deap.tools.cxTwoPoint)  # 两点交叉
        toolbox.register("mutate", deap.tools.mutUniformInt, low=0, up=self.Nr, indpb=0.1)  # 均匀变异
        toolbox.register("select", deap.tools.selTournament, tournsize=3)  # 锦标赛选择

        population = toolbox.population(n=500)  # 种群大小100

        self.start = time.time()
        deap.algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=1000, verbose=True)

        # best
        best_individual = deap.tools.selBest(population, k=1)[0]
        _ = self.objective_function(best_individual)

        arr = np.array(self.logger)
        np.savetxt(self.save_dir+"ga_logger.txt", arr, fmt='%.3e')
        np.savetxt(self.save_dir+"chiplet_allocation.txt", np.array(best_individual).reshape(1, -1), fmt='%d')
        # print(arr)
        plt.plot(arr[:, 1], arr[:, -1], "-b")
        plt.plot(arr[:, 1], np.minimum.accumulate(arr[:, -1]), "-r")
        plt.savefig(self.save_dir+"ga_converge_curve.png")
        print("最优分配:", best_individual)

    