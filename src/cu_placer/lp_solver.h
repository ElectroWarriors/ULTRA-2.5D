#include <ilcplex/ilocplex.h>
#include <iostream>
#include <vector>
#include <thread>

#include "type.h"
#include "db.h"

enum class ConType { LessEq, GreaterEq, Equal };
typedef std::pair<IndexType, LocType> VarIdCoeffPairType;

class LP_Solver {
public:
    // 构造函数：初始化CPLEX环境和模型
    LP_Solver() : _env(), _model(_env), _cplex(_model) {
        _cplex.setOut(_env.getNullStream()); // 关闭冗余输出
        _cplex.setParam(IloCplex::Threads, std::thread::hardware_concurrency()); // 启用多线程支持
        _cplex.setParam(IloCplex::Param::Parallel, 1); // 1: 自动选择, -1: 顺序模式, 0: 确定性并行
    }

    const IloEnv& env() const { return _env; }

    void addVariables(IndexType numVars) {
        _vars = IloNumVarArray(_env);
        for (IndexType i = 0; i < numVars; ++i) {
            IloNumVar var(_env, 0.0, IloInfinity, ILOFLOAT);
            _vars.add(var);
            _model.add(var);  // 每个变量单独添加到模型中
        }
    }

    IloNumVar& var(IndexType i) { return _vars[i]; }

    void setInitialSolution(const std::vector<RealType>& initVals) {
        IloNumVarArray startVars(_env);
        IloNumArray startVals(_env);

        for (IndexType i = 0; i < initVals.size(); ++i) {
            startVars.add(_vars[i]);
            startVals.add(initVals[i]);
        }

        _cplex.addMIPStart(startVars, startVals);

    }

    // 设置目标函数（最大化）
    void setMinimizeObjective(IloExpr obj) { _model.add(IloMinimize(_env, obj)); }

    void LPaddConstraint(IloExpr expr, ConType lgeq, RealType obj) {
        switch (lgeq) {
            case ConType::LessEq   : _model.add(expr <= obj); break;
            case ConType::GreaterEq: _model.add(expr >= obj); break;
            case ConType::Equal    : _model.add(expr == obj); break;
            default                : break;
        }
    }

IloNumVar addAbsConstraint(IloExpr diff) {
    // IloExpr diff1(_env); diff1 += (exprA - exprB);
    // IloExpr diff2(_env); diff2 += (exprB - exprA);

    // IloNumVar d(_env, 0.0, IloInfinity, ILOFLOAT);
    // _model.add(d);

    // _model.add(d >= diff1); _model.add(d >= diff2);

    // diff1.end(); diff2.end();

    // return d;
    IloNumVar d(_env, 0.0, IloInfinity, ILOFLOAT);
    _model.add(d); _model.add(d >= diff); _model.add(d >= -diff);
    diff.end();
    return d;
}


    bool solve() {
        try { return _cplex.solve(); } 
        catch (const IloException& e) { std::cerr << "CPLEX Error: " << e << std::endl; return false; }
    }

    RealType getObjectiveValue() const { return _cplex.getObjValue(); }

    RealType getVariableValue(IndexType i) const { return _cplex.getValue(_vars[i]); }

    ~LP_Solver() { _cplex.end(); _model.end(); _env.end(); }

private:
    IloEnv _env;          // CPLEX环境
    IloModel _model;      // 优化模型
    IloCplex _cplex;      // 求解器实例
    IloNumVarArray _vars; // 变量数组
};

