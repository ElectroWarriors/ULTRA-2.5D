#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <cuda_runtime.h>
#include"type.h"
#include"gplacer.h"
#include"db.h"
#include"CGLegalizer.h"

#define OPTIM_ENABLE_ARMA_WRAPPERS
#define OPTIM_ENABLE_GD_METHODS
#include"armadillo"
#include"optim.hpp"

using namespace db;


namespace gp {

IndexType HpwlCalculator::_next_id = 0;
IndexType BdyCalculator::_next_id  = 0;
IndexType OvlCalculator::_next_id  = 0;
IndexType CraCalculator::_next_id  = 0;
IndexType GPlacer::_iter = 0;


void GPlacer::ParametersInit(bool en_opt, bool en_calc, bool en_var, bool is_var_rand) {
    if (en_opt)  { OptimParametersInit(); }
    if (en_calc) { OperatorInit(); }
    if (en_var)  { VariableInit(is_var_rand); }
    if (utils::EN_GPU) { CUDAInit(); }
}

void GPlacer::OptimParametersInit() {
    spdlog::info("[nlp] parameters init");
    _hpwl_params[0] = utils::HPWL_PARAM0; // gamma
    _hpwl_params[1] = utils::HPWL_PARAM1_XY; // weight
    _hpwl_params[2] = utils::HPWL_PARAM1_ALPHA; // weight

    _bdy_params[0]  = utils::BDY_PARAM0; // gamma
    _bdy_params[1]  = utils::BDY_PARAM1; // weight

    _ovl_params[0]  = utils::OVL_PARAM0; // gamma
    _ovl_params[1]  = utils::OVL_PARAM1; // weight

    _cra_params[0]  = utils::CRA_PARAM0; // gamma
    _cra_params[1]  = utils::CRA_PARAM1; // weight
}

void GPlacer::OperatorInit() {
    spdlog::info("[nlp] calculators init");
    OperatorClear();
    // hpwl (connectivity)
    for (auto& ptr_cpair : _db.cpairs()) {
        CPairType* cpair = ptr_cpair.get();
        // _v_hpwl_calc.emplace_back(HpwlCalculator(HpwlInput(cpair->cluster0(), cpair->cluster1()), _hpwl_params));
        addHpwlCalc(cpair->cluster0(), cpair->cluster1());
    }
    // boundary
    for (auto& ptr_chiplet_i : _db.chiplets()) {
        Chiplet* chiplet_i = ptr_chiplet_i.get();
        Box& bdybox = chiplet_i->boundaryBox();
        addBdyCalc(chiplet_i, bdybox.point(), bdybox.size());
    }

    // overlap between chiplet (cut sume situation)
    for (auto chiplet_i : _db.getReticlewithId(0)->vchiplets()) {
        for (auto reticle : chiplet_i->cover_reticles()) {
            if (reticle->id() == 0) { std::cout << "[Warning] two chiplets both in stitching region have connection! " << std::endl; }
            for (auto chiplet_j : reticle->vchiplets()) {
                // _v_ovl_calc.emplace_back(OvlCalculator(OvlInput(chiplet_i, chiplet_j), _ovl_params));
                addOvlCalc(chiplet_i, chiplet_j);
            }
        }
    }
    for (auto& ptr_reticle : _db.reticles()) {
        Reticle* reticle = ptr_reticle.get();
        for (IndexType i=0; i < reticle->vchiplets().size(); i++) {
            Chiplet* chiplet_i = reticle->vchiplets()[i];
            for (IndexType j=i+1; j < reticle->vchiplets().size(); j++) {
                Chiplet* chiplet_j = reticle->vchiplets()[j];
                // _v_ovl_calc.emplace_back(OvlCalculator(OvlInput(chiplet_i, chiplet_j), _ovl_params));
                addOvlCalc(chiplet_i, chiplet_j);
            }
        }
    }
    // cross reticle chiplet need to have enough space overlap with reticle
    for (auto& ptr_chiplet_i : _db.chiplets()) {
        Chiplet* chiplet_i = ptr_chiplet_i.get();
        if (chiplet_i->reticle_id() != 0) { continue; }
        std::unordered_map<Reticle*, LocType> demandAmin_map;
        for (ClusterType* cluster : chiplet_i->vclusters()) {
            Chiplet* chiplet_j = cluster->connCluster()->chiplet();
            if (chiplet_j->cover_reticles().size() > 1 && (chiplet_i->is_newgen() == false && chiplet_j->is_newgen() == false)) {
                // temperary solution, later will be update to let previous chiplet should not be cross reticle chiplet
                throw std::invalid_argument("an error exist on `OperatorInit`");
            }
            demandAmin_map[chiplet_j->cover_reticles()[0]] += cluster->pin_area();
        }
        for (auto& rA : demandAmin_map) {
            // _v_cra_calc.emplace_back(CraCalculator(CraInput(chiplet_i, rA.first, rA.second), _cra_params));
            addCraCalc(chiplet_i, rA.first, rA.second);
        }
    }

    if (utils::EN_DEBUG && utils::EN_OUTPUT_OP) { print_operator_msg(_workdir+_casedir); }
}

void GPlacer::VariableInit(bool is_random) {
    IndexType circ = 2 + utils::NUM_DISCRETE_VAR;
    _x_init.set_size(_db.chiplets().size()*circ);
    IndexType idx = 0; 

    if (is_random) {
        std::random_device rd;
        std::mt19937 gen(rd());

        for (auto& opbdy : _v_bdy_calc) { // 顺序同chiplet
            Box box = opbdy.box();

            std::normal_distribution<RealType> dist_x(box.xc(), box.w() / 10.0);
            std::normal_distribution<RealType> dist_y(box.yc(), box.h() / 10.0);

            auto z = _db.chiplets()[idx].get()->z();

            _x_init(idx*circ+0) = dist_x(gen);
            _x_init(idx*circ+1) = dist_y(gen);

            for (IndexType i=0; i<circ-2; i++) { _x_init(idx*circ+2+i) = z[i]; }

            idx++;
        }
    } else {
        for (auto& ptr_chiplet_i : _db.chiplets()) {
            Chiplet* chiplet_i = ptr_chiplet_i.get();

            _x_init(idx*circ+0) = chiplet_i->xc();
            _x_init(idx*circ+1) = chiplet_i->yc();

            auto z = _db.chiplets()[idx].get()->z();
            for (IndexType i=0; i<circ-2; i++) { _x_init(idx*circ+2+i) = z[i]; }

            idx++;
        }
    }
}

void GPlacer::CUDAInit() {
    spdlog::info("[nlp] cuda init");
    if (!utils::EN_GPU) { return; }
    // chiplet
    std::vector<CUDA_Chiplet> host_chiplets;
    for (auto& ptr_chiplet : _db.chiplets()) { host_chiplets.emplace_back(CUDA_Chiplet(ptr_chiplet.get())); }
    // calculator
    std::vector<CUDA_HpwlData> host_hpwldatas;
    std::vector<CUDA_BdyData> host_bdydatas;
    std::vector<CUDA_OvlData> host_ovldatas;
    std::vector<CUDA_CraData> host_cradatas;
    for (auto& hpwl_calc : _v_hpwl_calc) { hpwl_calc.cuda_hpwldata_generate(); host_hpwldatas.emplace_back(hpwl_calc.cuda_hpwl_data()); }
    for (auto& bdy_calc  : _v_bdy_calc)  {  bdy_calc.cuda_bdydata_generate();   host_bdydatas.emplace_back(bdy_calc.cuda_bdy_data()); }
    for (auto& ovl_calc  : _v_ovl_calc)  {  ovl_calc.cuda_ovldata_generate();   host_ovldatas.emplace_back(ovl_calc.cuda_ovl_data()); }
    for (auto& cra_calc  : _v_cra_calc)  {  cra_calc.cuda_cradata_generate();   host_cradatas.emplace_back(cra_calc.cuda_cra_data()); }
    // hyper-param
    CUDA_Param host_params;
    host_params.setparams(_hpwl_params, _bdy_params, _ovl_params, _cra_params);
    
    _cuda_host_data.num_chiplets  = _db.chiplets().size();
    _cuda_host_data.num_hpwldatas = _v_hpwl_calc.size();
    _cuda_host_data.num_bdydatas  = _v_bdy_calc.size();
    _cuda_host_data.num_ovldatas  = _v_ovl_calc.size();
    _cuda_host_data.num_cradatas  = _v_cra_calc.size();
    _cuda_host_data.num_iodata    = _x_init.size();

    CUDA_CHECK(cudaMalloc(&_cuda_host_data.chiplets      , _cuda_host_data.num_chiplets       * sizeof(CUDA_Chiplet)));
    CUDA_CHECK(cudaMalloc(&_cuda_host_data.hpwldatas     , _cuda_host_data.num_hpwldatas      * sizeof(CUDA_HpwlData)));
    CUDA_CHECK(cudaMalloc(&_cuda_host_data.bdydatas      , _cuda_host_data.num_bdydatas       * sizeof(CUDA_BdyData)));
    CUDA_CHECK(cudaMalloc(&_cuda_host_data.ovldatas      , _cuda_host_data.num_ovldatas       * sizeof(CUDA_OvlData)));
    CUDA_CHECK(cudaMalloc(&_cuda_host_data.cradatas      , _cuda_host_data.num_cradatas       * sizeof(CUDA_CraData)));
    CUDA_CHECK(cudaMalloc(&_cuda_host_data.params        , _cuda_host_data.num_params         * sizeof(CUDA_Param)));
    CUDA_CHECK(cudaMalloc(&_cuda_host_data.param_updaters, _cuda_host_data.num_param_updaters * sizeof(CUDA_ParamUpdater)));
    CUDA_CHECK(cudaMalloc(&_cuda_host_data.x_in          , _cuda_host_data.num_iodata         * sizeof(RealType)));
    CUDA_CHECK(cudaMalloc(&_cuda_host_data.grad          , _cuda_host_data.num_iodata         * sizeof(RealType)));
    

    CUDA_CHECK(cudaMemcpy(_cuda_host_data.chiplets , host_chiplets.data(),  _cuda_host_data.num_chiplets  * sizeof(CUDA_Chiplet) , cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(_cuda_host_data.hpwldatas, host_hpwldatas.data(), _cuda_host_data.num_hpwldatas * sizeof(CUDA_HpwlData), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(_cuda_host_data.bdydatas , host_bdydatas.data(),  _cuda_host_data.num_bdydatas  * sizeof(CUDA_BdyData) , cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(_cuda_host_data.ovldatas , host_ovldatas.data(),  _cuda_host_data.num_ovldatas  * sizeof(CUDA_OvlData) , cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(_cuda_host_data.cradatas , host_cradatas.data(),  _cuda_host_data.num_cradatas  * sizeof(CUDA_CraData) , cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(_cuda_host_data.params   , &host_params        ,  _cuda_host_data.num_params    * sizeof(CUDA_Param)   , cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(_cuda_host_data.x_in     , _x_init.memptr()    ,  _cuda_host_data.num_iodata    * sizeof(RealType)     , cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&_cuda_device_data, sizeof(CUDA_DataPackage)));
    CUDA_CHECK(cudaMemcpy( _cuda_device_data, &_cuda_host_data, sizeof(CUDA_DataPackage), cudaMemcpyHostToDevice));
}



void GPlacer::AdamOptimize(arma::vec& x_out, IndexType stage) {
    // strategy.json stores report-scale targets (report qwl_* uses hpwl_* here).
    static const std::unordered_map<std::string, double> target_scale_factors = {
        {"hpwl_smoothness", 10.0}, {"ovl_smoothness", 10.0}, {"cra_smoothness", 10.0},
        {"hpwl_w_xy", 100.0}, {"hpwl_w_rot", 100.0}, {"hpwl_w_flip", 100.0},
        {"bdy_w_xy", 300.0}, {"ovl_w_xy", 100.0}, {"cra_w_xy", 100.0}
    };
    spdlog::info("[nlp] optimization setting init");
    x_out = _x_init;
    auto objfunc = std::bind(&GPlacer::ObjFunc, this, 
                             std::placeholders::_1, 
                             std::placeholders::_2, 
                             std::placeholders::_3);

    optim::algo_settings_t settings;
    settings.gd_settings.method = 6; // Adam
    settings.gd_settings.par_step_size = 0.5;
    // settings.iter_max = _max_iter;
    // settings.rel_sol_change_tol = 1E-3;
    // if (utils::EN_DEBUG && utils::EN_OUTPUT_GD) {settings.print_level = 1; } else { settings.print_level = 0; }
    
    bool finish;
    spdlog::info("[nlp] setting stretegy {}", stage);
    IndexType substage = 0;
    for (auto strategy_i : _json_strategy) {
        spdlog::info("[nlp] stage(stretegy) {}: substage {}", stage, substage++);
        _max_iter = strategy_i["max_iter"].asInt();
        settings.iter_max = _max_iter;

        // Restore internal units on this stage's copy for both CPU and GPU.
        for (auto& updater : strategy_i["updaters"]) {
            auto it = target_scale_factors.find(updater["param"].asString());
            if (it != target_scale_factors.end()) {
                updater["target"] = updater["target"].asDouble() * it->second;
            }
        }

        if (utils::EN_GPU) {
            if (strategy_i["stage"].asInt() != 0) { continue; } 
            CUDA_ParamUpdater updater_i;
            for (auto param_i : strategy_i["updaters"]) {
                auto it = _cuda_param_map.find(param_i["param"].asString());
                if (it != _cuda_param_map.end()) {
                    auto [row, col] = it->second;
                    updater_i.enable_at(row, col) = true;
                    updater_i.target_at(row, col) = param_i["target"].asDouble();
                    updater_i.mul_at(row, col)    = param_i["decay"].asDouble();
                }
            }
            CUDA_CHECK(cudaMemcpy(_cuda_host_data.param_updaters, &updater_i, sizeof(CUDA_ParamUpdater), cudaMemcpyHostToDevice));
        } else {
            if (strategy_i["stage"].asInt() != stage) { continue; }
            
            utils::USE_GUMBEL = strategy_i["use_gumbel"].asBool();
            utils::USE_FLIP   = strategy_i["use_flip"].asBool();
            _pupdaters.clear();

            for (auto updater : strategy_i["updaters"]) {
                auto it = _param_map.find(updater["param"].asString());
                if (it != _param_map.end()) {
                    _pupdaters.emplace_back(GPParamUpdater(it->second, updater["target"].asDouble(), updater["decay"].asDouble()));
                }
            }
        }
        finish = optim::gd(x_out, objfunc, nullptr, settings);
        // launchChipletPrintKernel(_cuda_device_data);

    }
    
    if (utils::EN_GPU) { CUDA_UpdateHostChiplet(); }
    _print_geometric_msg(_workdir+_casedir, _iter++);
    // if (utils::USE_FLIP) { GreedyOptimization(); }
    GreedyOptimization();
    GreedyOptimization();

    _print_geometric_msg(_workdir+_casedir, _iter++);

}

void GPlacer::BfgsOptimize() {

}

double GPlacer::ObjFunc(const arma::vec& x, arma::vec* grad_out, void* opt_data) {
    IndexType circ = 2 + utils::NUM_DISCRETE_VAR;
    IndexType n_boxes = x.n_elem / circ;
    RealType value = 0;
    
    if (_iter % 1000 == 1) { spdlog::info("iter: {}", _iter); }

    if (utils::EN_GPU) {
        
        value = GpuValueGradientCalculation(x, grad_out);

    } else {
        for (IndexType i=0; i<n_boxes; i++) {
            Chiplet* chiplet_i = _db.chiplets()[i].get();
            chiplet_i->setLoc(x(circ*i), x(circ*i+1));

            std::array<RealType, utils::NUM_DISCRETE_VAR> x_z;
            for (IndexType j=0; j<utils::NUM_DISCRETE_VAR; j++) { x_z[j] = x(circ*i+2+j); }
            chiplet_i->setz(x_z);
            chiplet_i->z2theta(_hpwl_params[0], _hpwl_params[3]);
        }
        if (utils::EN_DEBUG && utils::EN_OUTPUT_GEO) { _print_geometric_msg(_workdir+_casedir, _iter); }

        ValueCalculation();
        GradientCalculation();
        value = ValueAccumulation();

        if (grad_out) {
            GradientAccumulation();
            ParametersUpdate();
            if (utils::EN_DEBUG && utils::EN_OUTPUT_GEO) { _print_geometric_msg(_workdir+_casedir, _iter); }
            if (utils::EN_DEBUG && utils::EN_OUTPUT_GD) {print_val_grad_msg(_workdir+_casedir, _iter); }

            for (IndexType i=0; i<n_boxes; i++) {
                Chiplet* chiplet_i = _db.chiplets()[i].get();
                for (IndexType j=0; j<circ; j++) { (*grad_out)(circ*i+j) = chiplet_i->grad()[j]; }
            }
        }
        // if (_iter % 800 == 799) { GreedyOptimization(); }
    }

    // std::cout << "value = " << value << std::endl;
    // std::cout << "grad = [";
    // for (IndexType i = 0; i < grad_out->n_elem; ++i) {
    //     std::cout << (*grad_out)(i);
    //     if (i < grad_out->n_elem - 1) std::cout << ", ";
    // }
    // std::cout << "]" << std::endl;

    _iter++; 
    return value;
}

void GPlacer::ValueCalculation() {
    for (auto& ophpwl : _v_hpwl_calc) { ophpwl.evaluate_impl(); }
    for (auto& opbdy  : _v_bdy_calc ) { opbdy.evaluate_impl();  }
    for (auto& opovl  : _v_ovl_calc ) { opovl.evaluate_impl();  }
    for (auto& opcra  : _v_cra_calc ) { opcra.evaluate_impl();  }
}

void GPlacer::GradientCalculation() {
    for (auto& ophpwl : _v_hpwl_calc) { ophpwl.gradient_impl(); }
    for (auto& opbdy  : _v_bdy_calc ) { opbdy.gradient_impl();  }
    for (auto& opovl  : _v_ovl_calc ) { opovl.gradient_impl();  }
    for (auto& opcra  : _v_cra_calc ) { opcra.gradient_impl();  }
}

RealType GPlacer::ValueAccumulation() {
    RealType value = 0;
    for (auto& ophpwl : _v_hpwl_calc) { value+= _hpwl_params[1] * ophpwl.value(); }
    for (auto& opbdy  : _v_bdy_calc ) { value+= _bdy_params[1]  * opbdy.value();  }
    for (auto& opovl  : _v_ovl_calc ) { value+= _ovl_params[1]  * opovl.value();  }
    for (auto& opcra  : _v_cra_calc ) { value+= _cra_params[1]  * opcra.value();  }
    return value;
}

void GPlacer::GradientAccumulation() {
    for (auto& c : _db.chiplets()) { c.get()->clear_grad(); }
    for (auto& ophpwl : _v_hpwl_calc) { ophpwl.accumulate_gradient_impl(); }
    for (auto& opbdy  : _v_bdy_calc ) {  opbdy.accumulate_gradient_impl();  }
    for (auto& opovl  : _v_ovl_calc ) {  opovl.accumulate_gradient_impl();  }
    for (auto& opcra  : _v_cra_calc ) {  opcra.accumulate_gradient_impl();  }
}

RealType GPlacer::GpuValueGradientCalculation(const arma::vec& x_in, arma::vec* grad_out) {
    RealType value;
    CUDA_CHECK(cudaMemcpy(_cuda_host_data.x_in, x_in.memptr(), x_in.size() * sizeof(RealType), cudaMemcpyHostToDevice));
    // spdlog::info("update gpu chiplet");
    launchChipletUpdateKernel(_cuda_device_data, _cuda_host_data.num_chiplets);
    CUDA_CHECK(cudaDeviceSynchronize()); // wait for calculation
    // spdlog::info("calculate gradient and value");
    launchHpwlKernel(_cuda_device_data, _cuda_host_data.num_hpwldatas);
    launchBdyKernel(_cuda_device_data, _cuda_host_data.num_bdydatas);
    launchOvlKernel(_cuda_device_data, _cuda_host_data.num_ovldatas);
    launchCraKernel(_cuda_device_data, _cuda_host_data.num_cradatas);
    launchGradientClearKernel(_cuda_device_data, _cuda_host_data.num_iodata);
    CUDA_CHECK(cudaDeviceSynchronize());
    // spdlog::info("accumulate gradient and value");
    launchHpwlGradAccumulateKernel(_cuda_device_data, _cuda_host_data.num_hpwldatas);
    launchBdyGradAccumulateKernel(_cuda_device_data, _cuda_host_data.num_bdydatas);
    launchOvlGradAccumulateKernel(_cuda_device_data, _cuda_host_data.num_ovldatas);
    launchCraGradAccumulateKernel(_cuda_device_data, _cuda_host_data.num_cradatas);
    launchValueAccumulateKernel(_cuda_device_data);
    launchParamUpdateKernel(_cuda_device_data);
    CUDA_CHECK(cudaDeviceSynchronize());
    // spdlog::info("download gradient");
    CUDA_CHECK(cudaMemcpy(&value, &_cuda_device_data->value, sizeof(RealType), cudaMemcpyDeviceToHost));
    if (grad_out) {
        IndexType grad_size = _cuda_host_data.num_chiplets * (2 + utils::NUM_DISCRETE_VAR);
        CUDA_CHECK(cudaMemcpy(grad_out->memptr(), _cuda_host_data.grad, grad_size * sizeof(RealType), cudaMemcpyDeviceToHost));
    }
    // spdlog::info("finish");
    return value;

}

void GPlacer::CUDA_UpdateHostChiplet() {
    auto& db_chiplets = _db.chiplets();
    std::vector<CUDA_Chiplet> host_chiplets(_cuda_host_data.num_chiplets);
    CUDA_CHECK(cudaMemcpy(host_chiplets.data(),  _cuda_host_data.chiplets,  _cuda_host_data.num_chiplets  * sizeof(CUDA_Chiplet), cudaMemcpyDeviceToHost));
    for (IndexType i=0; i<host_chiplets.size(); i++) {
        auto db_chiplets_i = db_chiplets[i].get();
        db_chiplets_i->xc() = host_chiplets[i].xc(); db_chiplets_i->yc() = host_chiplets[i].yc();
        db_chiplets_i->theta() = host_chiplets[i].theta(); db_chiplets_i->theta_f() = host_chiplets[i].theta_f();
        for (IndexType j=0; j<utils::NUM_DISCRETE_VAR; j++) { db_chiplets_i->z(j) = host_chiplets[i].z(j); }
    }
}

void GPlacer::GreedyOptimization() { 
    for (auto& ptr_chiplet : _db.chiplets()) {
        Chiplet* chiplet_i = ptr_chiplet.get();
        RealType min_hpwl = REAL_TYPE_MAX; IndexType min_hpwl_idx = 0;
        for (IndexType idx=0; idx<utils::NUM_DISCRETE_VAR; idx++) {
            // if (!(utils::USE_FLIP) && idx >= 4) { continue; }
            chiplet_i->theta()   = utils::LEGAL_THETA[idx];
            chiplet_i->theta_f() = utils::LEGAL_THETA_F[idx];
            RealType curr_hpwl = 0;
            for (ClusterType* cluster_0 : chiplet_i->vclusters()) {
                ClusterType* cluster_1 = cluster_0->connCluster();
                Chiplet* chiplet_1 = cluster_1->chiplet();
                Point p0 = chiplet_i->point() + cluster_0->rot_offset_center();
                Point p1 = chiplet_1->point() + cluster_1->rot_offset_center();
                curr_hpwl += (p0 - p1).norm1();
            }
            if (curr_hpwl < min_hpwl) { min_hpwl = curr_hpwl; min_hpwl_idx = idx; }
        }
        chiplet_i->theta()   = utils::LEGAL_THETA[min_hpwl_idx];
        chiplet_i->theta_f() = utils::LEGAL_THETA_F[min_hpwl_idx];
        for (IndexType idx=0; idx<utils::NUM_DISCRETE_VAR; idx++) { chiplet_i->z(idx) = idx == min_hpwl_idx ? 1 : 0; }
        _print_geometric_msg(_workdir+_casedir, _iter++);
    }
}

void GPlacer::ParametersUpdate() {
    for (GPParamUpdater& pupdater : _pupdaters) { pupdater.update(); }
    if (utils::EN_DEBUG && utils::EN_OUTPUT_HP) {
        _param_logger->info("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}", \
                            _iter, _hpwl_params[0], _bdy_params[0], _ovl_params[0], _cra_params[0], \
                            _hpwl_params[1], _hpwl_params[2], _bdy_params[1], _ovl_params[1], _cra_params[1]);
    }
}

bool GPlacer::CustomizationChiplet(LocMatrix adj) {
    bool is_added_chiplet = false;
    // auto& ptr_cpair : _db.cpairs()
    for (auto& ptr_clu : _db.clusters()) {
        ClusterType* clu = ptr_clu.get();
        // std::cout << "cluster v0: " << clu->xc() << " " << clu->yc() << std::endl;
        // std::cout << "cluster v1: " << clu->offset_center().first() << " " << clu->offset_center().second() << std::endl;
    }
    IndexType n_cpair = _db.cpairs().size();
    for (IndexType idx_cpair=0; idx_cpair<n_cpair; idx_cpair++) {
        CPairType* cpair_i = _db.cpairs()[idx_cpair].get();
        if (! cpair_i->is_critical()) { continue; }

        IndexType gr0_id = cpair_i->chiplet0()->reticle_id(), gr1_id = cpair_i->chiplet1()->reticle_id();
        if (gr0_id != 0 && gr1_id != 0 && gr0_id == gr1_id) { continue; }
        
        const Point& p0 = cpair_i->chiplet0()->point() + cpair_i->cluster0()->rot_offset_center();
        const Point& p1 = cpair_i->chiplet1()->point() + cpair_i->cluster1()->rot_offset_center();
        for (auto& ptr_reticle_i : _db.reticles()) {
            Reticle* reticle_i = ptr_reticle_i.get();
            if (reticle_i->id() == 0) { continue; }
            if (reticle_i->contains(p0)) { gr0_id = reticle_i->id(); }
            if (reticle_i->contains(p1)) { gr1_id = reticle_i->id(); }
            if (gr0_id * gr1_id != 0) { break; }
        }
        if (gr0_id == gr1_id) { continue; }
        
        is_added_chiplet = true;
        spdlog::info("add new cross reticle chiplets"); 
        VecLocPair reticle_path = getReticlePathPairs(gr0_id-1, gr1_id-1, adj);
        // add chiplets
        Size xy_per_pins_0 = cpair_i->chiplet0()->xy_per_pin(), xy_per_pins_1 = cpair_i->chiplet1()->xy_per_pin();
        Size xy_per_pins = xy_per_pins_0 < xy_per_pins_1 ? xy_per_pins_0 : xy_per_pins_1;
        IndexType num_pins = 1.5 * 2 * 2 * cpair_i->cluster0()->vpins().size(); // global p/d pins, cluster p/d pins, two cluster
        LocType c_area = num_pins * xy_per_pins.mul();
        LocType c_size = std::sqrt(c_area);
        

        for (auto pair : reticle_path) {
            Reticle* r0 = _db.getReticlewithId(pair.first+1);
            Reticle* r1 = _db.getReticlewithId(pair.second+1);
            IndexType n_chiplets = _db.chiplets().size();

            Chiplet* c_s = cpair_i->chiplet0(); ClusterType* cl_s = cpair_i->cluster0();
            Chiplet* c_t = cpair_i->chiplet1(); ClusterType* cl_t = cpair_i->cluster1();
            // new chiplet
            Chiplet* chiplet_new = _db.allocateChiplet(n_chiplets, "cell_"+std::to_string(n_chiplets));
            chiplet_new->is_newgen() = true;
            chiplet_new->reticle_id() = 0;
            chiplet_new->addCoverReticle(r0); chiplet_new->addCoverReticle(r1);
            chiplet_new->setLoc((r0->point()+r1->point())/2);
            chiplet_new->setSize(c_size, c_size);
            chiplet_new->setNumPins(c_size/xy_per_pins.first(), c_size/xy_per_pins.second());
            chiplet_new->boundaryBox().setLoc((r0->point() + r1->point()) / 2);
            chiplet_new->boundaryBox().setSize(r0->size());

            CPairType* cpair_new = _db.allocateCPair(c_s, chiplet_new);
            cpair_new->is_critical() = false;

            ClusterType* cl_new0 = _db.allocateCluster(chiplet_new);
            ClusterType* cl_new1 = _db.allocateCluster(chiplet_new);
            chiplet_new->addCluster(cl_new0); chiplet_new->addCluster(cl_new1);
            
            cpair_i->setChipletPair(chiplet_new, c_t);
            cpair_i->setClusterPair(cl_new1, cl_t);
            cl_new1->setConnCluster(cl_t);
            cl_new1->virtual_num_pins() = cl_t->vpins().size();
            cl_t->setConnCluster(cl_new1);

            cpair_new->setChipletPair(c_s, chiplet_new);
            cpair_new->setClusterPair(cl_s, cl_new0);
            cl_new0->setConnCluster(cl_s);
            cl_new0->virtual_num_pins() = cl_s->vpins().size();
            cl_s->setConnCluster(cl_new0);
            
            cl_new0->calculate_center();
            cl_new1->calculate_center();

            _db.findallChipletCoverReticles(chiplet_new->id());
        }
    }
    return is_added_chiplet;
}


VecLocPair GPlacer::getReticlePathPairs(IndexType id0, IndexType id1, const LocMatrix& adj) {
    const IndexType N = adj.n_rows;
    std::vector<IndexType> parent(N, -1);
    std::queue<IndexType> q;

    q.push(id0); parent[id0] = id0;

    while (!q.empty()) {
        IndexType u = q.front(); q.pop();
        for (IndexType v = 0; v < N; ++v) {
            if (adj(u, v) > 0 && parent[v] == -1) {
                parent[v] = u; q.push(v);
                if (v == id1) { break; }
            }
        }
    }

    VecLocPair path_pairs;

    if (parent[id1] != -1) {
        std::vector<IndexType> path;
        for (IndexType cur = id1; cur != id0; cur = parent[cur]) {
            path.push_back(cur);
        }
        path.push_back(id0);
        std::reverse(path.begin(), path.end());

        for (size_t i = 0; i + 1 < path.size(); ++i) {
            path_pairs.emplace_back(path[i], path[i + 1]);
        }
    } else {
        throw std::runtime_error("Reticle path unreachable between " + std::to_string(id0) + " and " + std::to_string(id1));
    }

    return path_pairs;
}



// ------------------------------------------------------------------------------------------------------------------------- //
void GPlacer::dbgtest_value_grad() {
    ParametersInit(true, true, true, true);
    _db.chiplets()[0].get()->point() = Point(70, 80);
    _print_geometric_msg(_workdir+_casedir, 10000);
    // Evaluate();
    GradientCalculation();
    print_val_grad_msg(_workdir+_casedir, 10000);
    
}


void GPlacer::print_operator_msg(NameType path) {
    NameType basepath = path+"placer/log_"+utils::LOG_TIME+"/calculator_init/";
    auto calc_logger = spdlog::basic_logger_mt("calc_logger", basepath+"calculator.txt");
    calc_logger->info("");
    // hpwl
    calc_logger->set_pattern("[%l] %v");
    calc_logger->info("there are {} hpwl calculator: ", _v_hpwl_calc.size());
    for (auto& ophpwl : _v_hpwl_calc) { 
        calc_logger->info("-- {} the witelength is calculated between chiplet {}(location: [{}, {}, {}], cluster {} with offset: [{}, {}]) and chiplet {}(location: [{}, {}, {}], cluster {} with offset: [{}, {}])", \
                           ophpwl.id(), \
                           ophpwl.chiplet0()->id(), ophpwl.chiplet0()->xc(), ophpwl.chiplet0()->yc(), ophpwl.chiplet0()->theta()*180/PI, ophpwl.cluster0()->id(), ophpwl.offset0().first(), ophpwl.offset0().second(), \
                           ophpwl.chiplet1()->id(), ophpwl.chiplet1()->xc(), ophpwl.chiplet1()->yc(), ophpwl.chiplet1()->theta()*180/PI, ophpwl.cluster1()->id(), ophpwl.offset1().first(), ophpwl.offset1().second());
    }
    // bdy
    calc_logger->info("");
    calc_logger->info("there are {} boundary calculator: ", _v_bdy_calc.size());
    for (auto& opbdy : _v_bdy_calc) {
        Box bdybox = opbdy.box();
        calc_logger->info("-- {} the boundary constraint of chiplet {}: xl={}, yl={}, xh={}, yh={}", opbdy.id(), opbdy.chiplet()->id(), \
                           bdybox.xc()-bdybox.w()/2, bdybox.yc()-bdybox.h()/2, bdybox.xc()+bdybox.w()/2, bdybox.yc()+bdybox.h()/2);
    }
    // ovl
    calc_logger->info("");
    calc_logger->info("there are {} overlap calculator: ", _v_ovl_calc.size());
    for (auto& opovl : _v_ovl_calc) {
        calc_logger->info("-- {} the overlap is calculated between chiplet {} and {}", opovl.id(), opovl.chiplet0()->id(), opovl.chiplet1()->id()); 
    }
    // cross reticle
    calc_logger->info("");
    calc_logger->info("there are {} minimum cross reticle area calculator: ", _v_cra_calc.size());
    for (auto& opcra : _v_cra_calc) {
        calc_logger->info("-- {} the chiplet {} should cover {} area in reticle {}", opcra.id(), opcra.chiplet()->id(), opcra.Amin(), opcra.reticle()->id()); 
    }

    calc_logger->flush();
    spdlog::drop("calc_logger");
}

void GPlacer::_print_geometric_msg(NameType path, IndexType idx) {
    NameType basepath = path+"placer/log_"+utils::LOG_TIME+"/nlp_iteration/";
    auto gomt_logger = spdlog::basic_logger_mt("gomt_logger", basepath+std::to_string(idx)+"_geo.txt");
    gomt_logger->set_pattern("%v");
    // chiplet
    gomt_logger->info("# Chiplet");
    for (auto& ptr_chiplet : _db.chiplets()) {
        Chiplet* ci = ptr_chiplet.get();
        gomt_logger->info("{} {} {} {} {} {} {}", ci->id(), ci->xc(), ci->yc(), ci->theta()*180/PI, ci->theta_f()*180/PI, ci->w(), ci->h());
    }
    gomt_logger->info("# end Chiplet");
    // cpair
    gomt_logger->info("# CPair");
    for (auto& ptr_cpair : _db.cpairs()) {
        CPairType* cpi = ptr_cpair.get();
        gomt_logger->info("{} {} {} {} {} {}", cpi->chiplet0()->id(), cpi->cluster0()->offset_center().first(), cpi->cluster0()->offset_center().second(), \
                                               cpi->chiplet1()->id(), cpi->cluster1()->offset_center().first(), cpi->cluster1()->offset_center().second());
    }
    gomt_logger->info("# end CPair");
    gomt_logger->flush();
    spdlog::drop("gomt_logger");
}



void GPlacer::print_val_grad_msg(NameType path, IndexType idx) {
    NameType basepath = path+"placer/log_"+utils::LOG_TIME+"/nlp_iteration/";
    auto iter_logger = spdlog::basic_logger_mt("iter_logger", basepath+"gd_"+std::to_string(idx)+".txt");
    iter_logger->info("");
    iter_logger->set_pattern("[%l] %v");
    // total chiplets grad
    iter_logger->info("");
    iter_logger->info("chiplet x, y, theta, z0~z3, grad_x, grad_y, grad_z0 ~ grad_z3");
    for (auto& ptr_chiplet : _db.chiplets()) {
        Chiplet* c = ptr_chiplet.get();
        iter_logger->info("chiplet {}: x={}, y={}, theta={}, z=[{}, {}, {}, {}] | dx={}, dy={}, dz=[{}, {}, {}, {}]", c->id(), \
                          c->xc(), c->yc(), c->theta(), c->z()[0], c->z()[1], c->z()[2], c->z()[3], \
                          c->grad()[0], c->grad()[1], c->grad()[2], c->grad()[3], c->grad()[4], c->grad()[5]);
    }
    // hpwl
    iter_logger->info("");
    iter_logger->info("there are {} hpwl calculator | gamma={}, weight_xy={}, weight_theta={}, gumbel_mul={}", \
                      _v_hpwl_calc.size(), _hpwl_params[0], _hpwl_params[1], _hpwl_params[2], _hpwl_params[3]);
    for (auto& ophpwl : _v_hpwl_calc) { 
        HpwlGrad grad0 = ophpwl.grad0(), grad1 = ophpwl.grad1();
        iter_logger->info("-- {} value={} | chiplet {}: grad=[{}, {}, {}, {}, {}, {}] | chiplet {}: grad=[{}, {}, {}, {}, {}, {}]", \
                          ophpwl.id(), ophpwl.value(), \
                          ophpwl.chiplet0()->id(), grad0[0], grad0[1], grad0[2], grad0[3], grad0[4], grad0[5], \
                          ophpwl.chiplet1()->id(), grad1[0], grad1[1], grad1[2], grad1[3], grad1[4], grad1[5]); 
    }
    // bdy
    iter_logger->info("");
    iter_logger->info("there are {} boundary calculator | gamma={}, weight={}", _v_bdy_calc.size(), _bdy_params[0], _bdy_params[1]);
    for (auto& opbdy : _v_bdy_calc) {
        Box bdybox = opbdy.box();
        iter_logger->info("-- {} value={} | chiplet {}: grad=[{}, {}]", opbdy.id(), opbdy.value(), opbdy.chiplet()->id(), opbdy.grad()[0], opbdy.grad()[1]);
    }
    // ovl
    iter_logger->info("");
    iter_logger->info("{} overlap calculator | gamma={}, weight={}", _v_ovl_calc.size(), _ovl_params[0], _ovl_params[1]);
    for (auto& opovl : _v_ovl_calc) {
        iter_logger->info("-- {} value: {} | chiplet {}: grad=[{}, {}] | chiplet {}: grad=[{}, {}]", \
                          opovl.id(), opovl.value(), \
                          opovl.chiplet0()->id(), opovl.grad0()[0], opovl.grad0()[1], opovl.chiplet1()->id(), opovl.grad1()[0], opovl.grad1()[1]); 
    }
    // cross reticle
    iter_logger->info("");
    iter_logger->info("{} minimum cross reticle area calculator | gamma={}, weight={}", _v_cra_calc.size(), _cra_params[0], _cra_params[1]);
    for (auto& opcra : _v_cra_calc) {
        iter_logger->info("-- {} value={} | ratget={} | chiplet {} in reticle {}: grad: [{}, {}]", opcra.id(), opcra.value(), opcra.Amin(), opcra.chiplet()->id(), opcra.reticle()->id(), \
                                                                                       opcra.grad()[0], opcra.grad()[1]); 
    }

    iter_logger->flush();
    spdlog::drop("iter_logger");
}





} // end namespace
