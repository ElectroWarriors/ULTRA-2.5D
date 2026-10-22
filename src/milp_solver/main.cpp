#include <ilcplex/ilocplex.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <workdir> <casedir>" << std::endl;
        return 1;
    }

    std::string workdir = argv[1];
    std::string casedir = argv[2];
    std::string filename;
    if (argc >= 4) { filename = argv[3]; } 
    else { filename = workdir + casedir + "hur_partition/milp_model.lp"; }

    IloEnv env;
    try {
        IloModel model(env);
        IloCplex cplex(env);

        IloObjective obj;
        IloNumVarArray vars(env);
        IloRangeArray rng(env);

        // load model
        cplex.importModel(model, filename.c_str(), obj, vars, rng);
        cplex.extract(model);

        if (!cplex.solve()) {
            std::cerr << "No solution found." << std::endl;
            env.end();
            return 1;
        }

        std::cout << "Obj = " << cplex.getObjValue() << std::endl;

        // 初始化分配结果（假设 chiplet 数量为最大索引 + 1）
        int Nc = 0;
        for (int i = 0; i < vars.getSize(); ++i) {
            std::string name = vars[i].getName();
            if (name.find("X(") == 0 && cplex.getValue(vars[i]) > 0.5) {
                size_t l = name.find("(");
                size_t m = name.find("_", l);
                size_t r = name.find(")", m);
                if (l == std::string::npos || m == std::string::npos || r == std::string::npos)
                    continue;
                try {
                    int chiplet = std::stoi(name.substr(l + 1, m - l - 1));
                    Nc = std::max(Nc, chiplet + 1);
                } catch (...) {}
            }
        }

        std::vector<int> allocation(Nc, -1);

        // 解析变量名并提取分配结果
        for (int i = 0; i < vars.getSize(); ++i) {
            std::string name = vars[i].getName();
            if (name.find("X(") != 0 || cplex.getValue(vars[i]) < 0.5)
                continue;

            size_t l = name.find("(");
            size_t m = name.find("_", l);
            size_t r = name.find(")", m);
            if (l == std::string::npos || m == std::string::npos || r == std::string::npos)
                continue;

            try {
                int chiplet = std::stoi(name.substr(l + 1, m - l - 1));
                int reticle = std::stoi(name.substr(m + 1, r - m - 1));
                allocation[chiplet] = reticle;
            } catch (std::exception& e) {
                std::cerr << "Failed to parse variable name: " << name << std::endl;
            }
        }

        // 输出分配结果
        std::ofstream fout(workdir + casedir + "hur_partition/chiplet_allocation.txt");
        for (int i = 0; i < allocation.size(); ++i) {
            fout << allocation[i] << " ";
        }
        fout << "\n";
        fout.close();

    } catch (IloException& e) {
        std::cerr << "CPLEX exception: " << e.getMessage() << std::endl;
    }

    env.end();
    return 0;
}
