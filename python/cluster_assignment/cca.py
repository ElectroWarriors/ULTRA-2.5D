import os
import numpy as np
import random
import matplotlib.pyplot as plt

class PinAssignment(object):
    def __init__(self, grid_size, cluster_ids, udlr_direction_id, save_dir, params=None):
        self.grid_size = grid_size # (50, 50)
        self.cluster_ids = cluster_ids
        self.udlr_direction_id = udlr_direction_id
        self.save_dir=save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        
        if (params is None):
            self.param_init_rand()
        else:
            self.param_init(params)
        self.grid_x, self.grid_y = np.meshgrid(np.arange(self.grid_size[0]), np.arange(self.grid_size[1]))
        self.threshold = 0.1
        self.base_lr = 0.1
        self.max_iters = 10000

    def bell_function(self, x, y, params):
        A, mu_x, mu_y, sigma_x, sigma_y, prefer_sigma_x, prefer_sigma_y = params[0:7]
        return A * np.exp(-((x - mu_x)**2 / (2 * (prefer_sigma_x*sigma_x)**2) + (y - mu_y)**2 / (2 * (prefer_sigma_y*sigma_y)**2)))
    
    # Initialize grid and parameters
    def param_init_rand(self):
        self.num_types = 4
        self.params = []
        for _ in range(self.num_types):
            # Randomly initialize parameters: center, width, amplitude
            mu_x = np.random.uniform(0, self.grid_size[0])
            mu_y = np.random.uniform(0, self.grid_size[1])
            sigma_x = np.random.uniform(1, self.grid_size[0] / 4)
            sigma_y = np.random.uniform(1, self.grid_size[1] / 4)
            prefer_sigma_x = 1
            prefer_sigma_y = 1
            A = np.random.uniform(1, 10)
            n_nets = np.random.uniform(1, 10)
            self.params.append([A, mu_x, mu_y, sigma_x, sigma_y, prefer_sigma_x, prefer_sigma_y, n_nets])
            self.target_counts.append(n_nets) # target_counts # [1000, 800, 300]

    
    def param_init(self, params):
        self.num_types = len(params)
        self.params = params
        self.target_counts = [param[7] for param in params] # target_counts # [1000, 800, 300]


    # Assign points to types based on bell function values
    def assign_types(self):
        cluster_ids = np.array(self.cluster_ids + [-1])
        values = np.array([self.bell_function(self.grid_x, self.grid_y, p) for p in self.params])
        max_values = np.max(values, axis=0)
        types = np.argmax(values, axis=0)
        types[max_values < self.threshold] = self.num_types  # Mark unoccupied points
        # print([np.sum(types == t) for t in range(self.num_types)])
        # print(types)
        self.types = cluster_ids[types] # [[cluster_ids[j] for j in i] for i in types]
        # print(self.types)
    
    def type_cut(self):
        for idx in range(len(self.udlr_direction_id)):
            udlr_id_vec = self.udlr_direction_id[idx]
            if (len(udlr_id_vec) <= 1):
                continue
            if (idx == 0 or idx == 1):
                lst = self.types[-1, :] if idx == 0 else self.types[0, :]
                all_change_idx = [i for i in range(1, len(lst)) if lst[i] != lst[i-1]]
                for change_idx in all_change_idx:
                    if (lst[change_idx] in udlr_id_vec and lst[change_idx-1] in udlr_id_vec):
                        self.types[:, :change_idx][self.types[:, :change_idx]==lst[change_idx]] = lst[change_idx-1]
                        self.types[:, change_idx:][self.types[:, change_idx:]==lst[change_idx-1]] = lst[change_idx]
            elif (idx == 2 or idx == 3):
                lst = self.types[:, 0] if idx == 2 else self.types[:, -1]
                all_change_idx = [i for i in range(1, len(lst)) if lst[i] != lst[i-1]]
                for change_idx in all_change_idx:
                    if (lst[change_idx] in udlr_id_vec and lst[change_idx-1] in udlr_id_vec):
                        self.types[:change_idx, :][self.types[:change_idx, :]==lst[change_idx]] = lst[change_idx-1]
                        self.types[change_idx:, :][self.types[change_idx:, :]==lst[change_idx-1]] = lst[change_idx]

    # Compute the type counts
    def compute_counts(self):
        self.counts = [np.sum(self.types == t) for t in self.cluster_ids]
    
    # Visualization function
    def visualize(self, iteration, chiplet_id, pass0_save1_show2=1):
        if (pass0_save1_show2 == 0):
            return
        plt.figure(figsize=(8, 8))
        plt.title(f"Iteration {iteration}")
        plt.imshow(self.types, origin='lower', cmap='tab20', extent=(0, self.grid_x.shape[1], 0, self.grid_x.shape[0]))
        plt.colorbar(label="Type")
        plt.xlabel("X")
        plt.ylabel("Y")
        # if self.counts:
        #     for t, count in enumerate(self.counts):
        #         plt.text(2, self.grid_x.shape[0] - 5 - t * 3, f"Type {t}: {count}", color="white", fontsize=10, ha="left", va="center", bbox=dict(facecolor='black', alpha=0.5))
        print(self.types)
        if (pass0_save1_show2 == 1):
            plt.savefig(self.save_dir+f"chiplet_{chiplet_id}_iteration_{iteration}.png")
        else:
            plt.show()

    # Update parameters with dynamic learning rate
    def update_params_dynamic(self):
        for i in range(self.num_types):
            # Compute error
            current_count = np.sum(self.types == self.cluster_ids[i])
            error = self.target_counts[i] - current_count
            dynamic_lr = self.base_lr * abs(error) / max(self.target_counts)  # Scale learning rate by error size
            # print(current_count, self.target_counts[i], error, dynamic_lr * np.sign(error))
            if error > 0 or error < -10:
                # Adjust the width (sigma_x, sigma_y)
                norm = self.params[i][3] / (self.params[i][3] + self.params[i][4])
                self.params[i][3] += (1-norm) * dynamic_lr * np.sign(error)
                self.params[i][4] += (  norm) * dynamic_lr * np.sign(error)

                # Adjust the center positions (mu_x, mu_y)
                # occupied_points = np.where(types == i)
                # if len(occupied_points[0]) > 0:
                #     avg_x = np.mean(grid_x[occupied_points])
                #     avg_y = np.mean(grid_y[occupied_points])
                #     params[i][1] += dynamic_lr * (avg_x - params[i][1]) * np.sign(error)
                #     params[i][2] += dynamic_lr * (avg_y - params[i][2]) * np.sign(error)

                # Adjust the amplitude (A)
                self.params[i][0] += dynamic_lr * np.sign(error)

    def run(self, chiplet_id):
        for iteration in range(self.max_iters):
            # Assign types
            self.assign_types()
            self.type_cut()
            # Compute current counts
            self.compute_counts()
            print(f"Iteration {iteration}: Counts = {self.counts}")

            # Visualize current state
            if (iteration % 500 == 0):
                self.visualize(iteration=iteration, chiplet_id=chiplet_id, pass0_save1_show2=1)

            # Check for convergence
            # if self.counts == self.target_counts:
            if all([1 if self.target_counts[i]<=self.counts[i]<=max(self.target_counts[i]*1.02, self.target_counts[i]+1) 
                    else 0 for i in range(len(self.target_counts))]):
                print("Converged!")
                self.visualize(iteration=iteration, chiplet_id=chiplet_id, pass0_save1_show2=1)
                break

            # Update parameters with dynamic learning rate
            self.update_params_dynamic()
        self.visualize(iteration="maxiter", chiplet_id=chiplet_id, pass0_save1_show2=1)
        if (False):
            # self.type_cut()
            self.visualize(iteration)
        return self.types

        
        # Fine-tune assignments to ensure exact match
        # types = fine_tune_assignments(grid_x, grid_y, types, params, target_counts)
        # final_counts = compute_counts(types, num_types)
        # print(f"Final Counts after Fine-tuning: {final_counts}")

        # Visualize the final result after fine-tuning
        # visualize(grid_x, grid_y, types, num_types, "Final (Fine-tuned)", counts=final_counts)
if __name__ == "__main__":
    grid_size = (100, 80)  # Grid dimensions
    target_counts = [500, 1000, 2000, 2000]  # Target counts for each type
    # A, mu_x, mu_y, sigma_x, sigma_y, prefer_sigma_x, prefer_sigma_y
    params = [[1, 0, 10, 50, 50, 1/4, 3/4, 500], 
              [1, 70, 0, 50, 50, 6/7, 1/7, 1000],
              [1, 100, 70, 50, 50, 1/5, 4/5, 2000], 
              [1, 30, 80, 50, 50, 7/8, 1/8, 2000]] 
    pa = PinAssignment(grid_size=grid_size, params=params)
    pa.run()

    