#include<iostream>
#include<string>

#include"spdlog/spdlog.h"

#include"utils.h"
#include"flow.h"
#include"db.h"

using namespace db;

void MainFlow::DatabaseInit() {
    spdlog::info("[database] init");
    // chiplets
    spdlog::info("[database] chiplets data init");
    for (IndexType chipletIdx=0; chipletIdx<_json_netlist["chiplets"].size(); chipletIdx++) {
        if (_json_netlist["chiplets"][chipletIdx]["id"].asInt() != _json_datadescript["chiplets"][chipletIdx]["id"].asInt()) {
            throw std::invalid_argument("there are some bug to be fixed");
        }
        auto js_chiplet_i = _json_netlist["chiplets"][chipletIdx];
        Chiplet* chiplet_i = _db.allocateChiplet(js_chiplet_i["id"].asInt(), js_chiplet_i["name"].asString());
        chiplet_i->is_newgen() = false;
        LocType xl = js_chiplet_i["loc"][0].asDouble() / utils::XSCALE,  yl = js_chiplet_i["loc"][1].asDouble() / utils::YSCALE;
        LocType w  = js_chiplet_i["size"][0].asDouble() / utils::XSCALE, h  = js_chiplet_i["size"][1].asDouble() / utils::YSCALE;
        chiplet_i->setLoc(xl + w/2, yl + h/2);
        chiplet_i->setSize(w, h);
        chiplet_i->setNumPins(_json_datadescript["chiplets"][chipletIdx]["numPin"][0].asInt(), 
                              _json_datadescript["chiplets"][chipletIdx]["numPin"][1].asInt());
        chiplet_i->space().fill(utils::WHITESPACE);
    }
    // pins
    spdlog::info("[database] pins data init");
    for (auto js_pin : _json_netlist["pins"]) {
        Pin* pin_i = _db.allocatePin(js_pin["id"].asInt(), js_pin["name"].asString());
    }
    // nets
    spdlog::info("[database] nets data init");
    for (auto js_net : _json_netlist["nets"]) {
        Net* net_i = _db.allocateNet(js_net["id"].asInt(), js_net["name"].asString());
    }
    // reticles
    spdlog::info("[database] reticles data init");
    for (IndexType i=0; i<=_v_reticle_msg.n_rows; i++) { // id=0: stitching region
        Reticle* reticle_i = _db.allocateReticle(i, "reticle_"+std::to_string(i));
        if (i == 0) { 
            reticle_i->setLoc(0, 0);
            reticle_i->setSize(0, 0);
        } else {
            LocType xl = 1.0*_v_reticle_msg(i-1, 0) / utils::XSCALE, yl = 1.0*_v_reticle_msg(i-1, 1) / utils::YSCALE; 
            LocType w  = 1.0*_v_reticle_msg(i-1, 2) / utils::XSCALE, h  = 1.0*_v_reticle_msg(i-1, 3) / utils::YSCALE;
            reticle_i->setLoc(xl + w / 2, yl + h / 2);
            reticle_i->setSize(w, h);
        }
    }

    // association
    spdlog::info("[database] generate association");
    spdlog::info("[database] chiplet variable: link pins");
    for (auto js_chiplet : _json_netlist["chiplets"]) {
        Chiplet* chiplet_i = _db.getChipletwithId(js_chiplet["id"].asInt());
        for (auto pinIdx : js_chiplet["pins_id"]) {
            Pin* pin = _db.getPinwithId(pinIdx.asInt());
            chiplet_i->addPin(pin);
        }
    }

    spdlog::info("[database] pin variable: link chiplets and nets.");
    for (auto js_pin : _json_netlist["pins"]) {
        Pin* pin_i = _db.getPinwithId(js_pin["id"].asInt());
        Chiplet* chiplet = _db.getChipletwithId(js_pin["cell_id"].asInt());
        Net* net = _db.getNetwithId(js_pin["net_id"].asInt());
        pin_i->setChiplet(chiplet);
        pin_i->setNet(net);
        LocType xLo = js_pin["ioShape"]["xLo"].asDouble() / utils::XSCALE, xHi = js_pin["ioShape"]["xHi"].asDouble() / utils::XSCALE;
        LocType yLo = js_pin["ioShape"]["yLo"].asDouble() / utils::YSCALE, yHi = js_pin["ioShape"]["yHi"].asDouble() / utils::YSCALE;
        LocType xc  = pin_i->owner_chiplet()->w()/2   , yc  = pin_i->owner_chiplet()->h()/2;
        pin_i->setLoc((xLo + xHi) / 2 - xc, (yLo + yHi) / 2 - yc);
        pin_i->setSize(xHi - xLo, yHi - yLo);
    }

    spdlog::info("[database] net variable: link pin");
    for (auto js_net : _json_netlist["nets"]) {
        Net* net_i = _db.getNetwithId(js_net["id"].asInt());
        for (auto pinIdx : js_net["pins_id"]) {
            Pin* pin = _db.getPinwithId(pinIdx.asInt());
            net_i->addPin(pin);
        }
    }
    // reticle <-> chiplet
    spdlog::info("[database] reticle/chiplet variable: link chiplet/reticle");
    for (IndexType i=0; i<_v_partition_msg.n_cols; i++) {
        Chiplet*  chiplet_i = _db.getChipletwithId(i);
        // spdlog::info("{}/{}, {}", i, _v_partition_msg.n_cols, _v_partition_msg(0, i));
        Reticle* reticle_i = _db.getReticlewithId(_v_partition_msg(0, i));
        chiplet_i->reticle_id() = _v_partition_msg(0, i);
        reticle_i->addChiplet(chiplet_i);
    }

    // generate cluster pair
    _db.conn_mat().set_size(_db.chiplets().size(), _db.chiplets().size());
    _db.weight_mat().set_size(_db.chiplets().size(), _db.chiplets().size());
    _db.conn_mat().zeros();
    _db.weight_mat().zeros();
    utils::readConnFromJSON(_db.conn_mat()  , _json_datadescript["connection_matrix"], _json_datadescript["use_sparse"].asBool()); 
    utils::readConnFromJSON(_db.weight_mat(), _json_datadescript["weight_matrix"]    , _json_datadescript["use_sparse"].asBool()); 
    spdlog::info("[database] generate cluster pair");
    for (auto& ptr_net : _db.nets()) {
        Net* net = ptr_net.get();
        if (net->id() == 0 || net->id() == 1 || net->vpins().size() < 2) {
            continue;
        } 
        std::vector<Pin*> pins = net->vpins();
        bool is_added_to_clusterpair = false;
        for (auto& ptr_cpair : _db.cpairs()) {
            CPairType* cpair = ptr_cpair.get();
            is_added_to_clusterpair = cpair->addPinPair(pins[0], pins[1]);
            if (is_added_to_clusterpair) { break; }
        }
        if (!is_added_to_clusterpair) {
            CPairType* cpair = _db.allocateCPair(pins[0]->owner_chiplet(), pins[1]->owner_chiplet());
            cpair->addPinPair(pins[0], pins[1]);
            cpair->is_critical() = (static_cast<IndexType>(_db.weight_mat()(pins[0]->owner_chiplet()->id(), pins[1]->owner_chiplet()->id())) == 9) ? true : false;
        }
    }
    
    // generate cluster
    spdlog::info("[database] generate cluster");
    for (auto& ptr_cpair : _db.cpairs()) {
        CPairType* cpair = ptr_cpair.get();
        ClusterType* cl0 = _db.allocateCluster(cpair->chiplet0(), cpair->pins0());
        ClusterType* cl1 = _db.allocateCluster(cpair->chiplet1(), cpair->pins1());
        cl0->calculate_center();
        cl1->calculate_center();
        cl0->calculate_wh();
        cl1->calculate_wh();
        cpair->chiplet0()->addCluster(cl0);
        cpair->chiplet1()->addCluster(cl1);
        cl0->setConnCluster(cl1);
        cl1->setConnCluster(cl0);
        cpair->setClusterPair(cl0, cl1);
    }

    spdlog::info("[database] chiplet variable: cross reticles");
    _db.findallChipletCoverReticles(INDEX_TYPE_MAX);

    // for (auto& ptr_chiplet_i : _db.chiplets()) {
    //     Chiplet* chiplet_i = ptr_chiplet_i.get();
    //     std::vector<Reticle*> v_reticles = chiplet_i->cover_reticles();
    //     if (chiplet_i->reticle_id()) {
    //         if (v_reticles.size() > 1) { std::cout << "error in boundary constraints generation" << std::endl; }
    //         // _v_bdy_calc.emplace_back(BdyCalculator(BdyInput(chiplet_i, v_reticles[0]->point(), v_reticles[0]->size()), _bdy_params));
    //         // addBdyCalc(chiplet_i, v_reticles[0]->point(), v_reticles[0]->size());
    //         chiplet_i->boundaryBox().setLoc(v_reticles[0]->point());
    //         chiplet_i->boundaryBox().setSize(v_reticles[0]->size());
    //     } else {
    //         Point pc = Point(0, 0);
    //         for (auto r : v_reticles) { pc = pc + r->point(); }
    //         pc = pc / v_reticles.size(); 
    //         Size size = v_reticles[0]->size();
    //         // _v_bdy_calc.emplace_back(BdyCalculator(BdyInput(chiplet_i, pc, size), _bdy_params));
    //         // addBdyCalc(chiplet_i, pc, size);
    //         chiplet_i->boundaryBox().setLoc(pc);
    //         chiplet_i->boundaryBox().setSize(size);
    //     }
    // }


    spdlog::info("[database] init success");

    if (utils::EN_OUTPUT_DB) {
        for (auto& ch : _db.chiplets()) { ch.get()->print_msg(_workdir+_casedir); }
        for (auto& re : _db.reticles()) { re.get()->print_msg(_workdir+_casedir); }
        for (auto& cp : _db.cpairs()  ) { cp.get()->print_msg(_workdir+_casedir); }
    }


}


void MainFlow::OutputNewLocation() {
    // std::ofstream file0(_workdir+_casedir+"placer/report_wl.txt");
    // std::ofstream file1(_workdir+_casedir+"placer/log_"+utils::LOG_TIME+"/report_wl.txt");
    // if (!file0.is_open() || !file1.is_open()) { std::cerr << "can not open file. " << std::endl; } 
    // else {
    //     IntType wirelength = 0;
    //     for (auto& cpair_i : _db.cpairs()) { wirelength += cpair_i.get()->v_paths()[0].size()*utils::XSCALE; }
    //     std::string str_wl = "wirelength: " + std::to_string(wirelength);
    //     file0 << str_wl;
    //     file1 << str_wl;
    // }

    for (IndexType chipletIdx=0; chipletIdx<_json_netlist["chiplets"].size(); chipletIdx++) {
        Chiplet* chiplet_i = _db.getChipletwithId(_json_netlist["chiplets"][chipletIdx]["id"].asInt());
        // spdlog::info("============== chiplet id = {} ==============", chiplet_i->id());
        // spdlog::info("norm:  xc={}, yc={}, w={}, h={}", chiplet_i->xc(), chiplet_i->yc(), chiplet_i->rot_w(), chiplet_i->rot_h());

        LocType xc = chiplet_i->xc()*utils::XSCALE, yc = chiplet_i->yc()*utils::YSCALE;
        LocType  w = chiplet_i->rot_real_w()      ,  h = chiplet_i->rot_real_h();

        // spdlog::info("scale: xc={}, yc={}, w={}, h={}", xc, yc, w, h);
        // spdlog::info(" ");

        Json::Value chiplet_i_loc(Json::arrayValue);
        chiplet_i_loc.append(xc - w / 2);
        chiplet_i_loc.append(yc - h / 2);
        _json_netlist["chiplets"][chipletIdx]["loc"] = chiplet_i_loc;

        Json::Value chiplet_i_size(Json::arrayValue);
        chiplet_i_size.append(w);
        chiplet_i_size.append(h);
        _json_netlist["chiplets"][chipletIdx]["size"] = chiplet_i_size;
    }

    for (IndexType pinIdx=0; pinIdx<_json_netlist["pins"].size(); pinIdx++) {
        Pin* pin_i = _db.getPinwithId(_json_netlist["pins"][pinIdx]["id"].asInt());
        Chiplet* chiplet_i = _db.getChipletwithId(_json_netlist["pins"][pinIdx]["cell_id"].asInt());
        LocType w = chiplet_i->rot_real_w(), h = chiplet_i->rot_real_h();

        Point pin_loc = pin_i->rot_flip_real_loc();
        Size pin_size = pin_i->real_size();
        Json::Value pin_i_loc(Json::arrayValue);
        pin_i_loc.append(w / 2 + pin_loc.first() );
        pin_i_loc.append(h / 2 + pin_loc.second());
        _json_netlist["pins"][pinIdx]["offset"] = pin_i_loc;
        _json_netlist["pins"][pinIdx]["ioShape"]["xLo"] = w / 2 + pin_loc.first()  - pin_size.first()  / 2;
        _json_netlist["pins"][pinIdx]["ioShape"]["xHi"] = w / 2 + pin_loc.first()  + pin_size.first()  / 2;
        _json_netlist["pins"][pinIdx]["ioShape"]["yLo"] = h / 2 + pin_loc.second() - pin_size.second() / 2;
        _json_netlist["pins"][pinIdx]["ioShape"]["yHi"] = h / 2 + pin_loc.second() + pin_size.second() / 2;
    }
    
    utils::WriteJSON(_workdir+_casedir+"placer/chiplet_pla.json", _json_netlist);
    utils::WriteJSON(_workdir+_casedir+"placer/log_"+utils::LOG_TIME+"chiplet_pla.json", _json_netlist);
}