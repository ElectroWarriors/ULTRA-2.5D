import time

import python.partition.partition as huristic_partition
import python.placer.placer as placer


# /***************************************************************************************************
#  *                                      编程荣辱观 · Coding Honor Code                              *
#  ***************************************************************************************************
#  *                                                                                                 *
#  * 以动手实践为荣，以只看不练为耻。       Honor in hands-on practice; shame in passive watching.      *
#  * 以打印日志为荣，以出错不报为耻。            Honor in logging; shame in silent failures.            *
#  * 以局部变量为荣，以全局变量为耻。           Honor in local scope; shame in global sprawl.           *
#  * 以自动测试为荣，以手工测试为耻。        Honor in automated testing; shame in manual poking.        *
#  * 以代码重用为荣，以复制粘贴为耻。               Honor in reuse; shame in copy-paste.                *
#  * 以多态应用为荣，以分支判断为耻。        Honor in polymorphism; shame in excessive branching.       *
#  * 以定义常量为荣，以魔法数字为耻。         Honor in named constants; shame in magic numbers.         *
#  * 以总结思考为荣，以不求甚解为耻。   Honor in reflection and understanding; shame in shallow coding. *
#  *                                                                                                 *
#  ***************************************************************************************************/


class MainFlow(object):
    def __init__(self, workdir, casedir):
        self.workdir = workdir
        self.casedir = casedir

    def partition(self, reticle_file, enmilp):
        reticle_file = self.workdir+"scripts/reticle_type/"+reticle_file
        hur = huristic_partition.Huristic_Partition(self.workdir, self.casedir, reticle_file)
        if (enmilp): hur.run_MIQP()
        else: hur.run_GA()

    def run_gplacer(self, option="gd"):
        pla = placer.Placer(self.workdir, self.casedir)
        if  (option == "gd")  : pla.gd_placer()
        elif(option == "milp"): pla.milp_placer()

if __name__=="__main__":
    # ---------------------- sweep cases ---------------------- #
    num_loop = 1
    work_path = f"/path/to/Multi_Reticle/"
    case_idx = [
        ["case_8c/", "reticle_2r_12000_25000.npz"], 
        ["case_22c/", "reticle_4r_25000_25000.npz"], 
        ["atplace_case/case_36c/", "reticle_2r_25000_15000.npz"], 
        ["atplace_case/case_44c/", "reticle_4r_30000_25000.npz"], 
        ["atplace_case/case_60c/", "reticle_4r_25000_25000.npz"], 
        ["case_225c/", "reticle_4r_24000_20000.npz"], 
        ["case_1225c/", "reticle_4r_24000_24000.npz"]
    ]
    for idx in range(len(case_idx)): # 
        file_path = f"celldata/{case_idx[idx][0]}"
        reticle   = case_idx[idx][1]
        for loop_idx in range(num_loop):
            flow = MainFlow(work_path, file_path)
            print("work space: " + work_path)
            print("file path : " + file_path)
            
            start = time.time()
            flow.partition(enmilp=True, reticle_file=reticle)
            flow.run_gplacer(option="gd")
            print(f"time test result: {time.time()-start}")

