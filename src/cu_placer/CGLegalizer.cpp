#include <iostream>
#include <algorithm>
#include <utility>

#include "CGLegalizer.h"
#include "lp_solver.h"

ConsEdges Legalizer::generateConsEdges(bool isHorG) {
    if (isHorG) { generateHorConsEdges(); return _hConsEdges; } 
    else        { generateVerConsEdges(); return _vConsEdges; }
}

void Legalizer::generateHorConsEdges() {
    _hCG.clear();
    _vCG.clear();
    _hConsEdges.clear();
    _vConsEdges.clear();
    // Init the irredundant constraint edges
    SweeplineCGGenerator sweepline(_db, _hConsEdges, _vConsEdges);
    sweepline.solve();
}

void Legalizer::generateVerConsEdges() {
    _hCG.clear();
    _vCG.clear();
    _hConsEdges.clear();
    _vConsEdges.clear();
    // Init the irredundant constraint edges
    SweeplineCGGenerator sweepline(_db, _hConsEdges, _vConsEdges);
    sweepline.solve();
}


bool Legalizer::legalize(IndexType direction) {
    // legalization
    auto phys2var_double = [&](IndexType chiplet_id, IndexType x0y1) -> IndexType  {
        return 2 * chiplet_id + x0y1;
    };
    auto phys2var_single = [&](IndexType chiplet_id, IndexType x0y1) -> IndexType  {
        return chiplet_id;
    };

    auto phys2var =  [&](IndexType chiplet_id, IndexType x0y1) -> IndexType  {
        if (direction == 0 || direction == 1) return phys2var_single(chiplet_id, x0y1);
        return phys2var_double(chiplet_id, x0y1);
    };
    
    spdlog::info("[legalization] start");

    for (auto& ptr_chiplet_i : _db.chiplets()) { ptr_chiplet_i.get()->legalize(); }

    LP_Solver solver;
    if (direction == 0 || direction == 1) { solver.addVariables(_db.chiplets().size()); } 
    else { solver.addVariables(2*_db.chiplets().size()); }

    IloExpr obj(solver.env()); 
    for (IndexType horver=0; horver<2; horver++) {
        if (direction == 0 && horver == 1) { continue; } // only horizontal: ignore vertical constraints -- fail
        if (direction == 1 && horver == 0) { continue; } // only vertical: ignore horizontal constraints -- fail
        bool isHor = horver == 0 ? true : false;

        // boundary
        for (auto& ptr_chiplet_i : _db.chiplets()) {
            Chiplet* chiplet_i = ptr_chiplet_i.get();
            
            IloNumVar var_i = solver.var(phys2var(chiplet_i->id(), horver));

            LocType  space_bdy = 0.5;
            Box&       bdy_box = chiplet_i->boundaryBox();
            RealType half_size = isHor ? chiplet_i->rot_w() / 2.0 : chiplet_i->rot_h() / 2.0;
            LocType bdy_Lo = isHor ? bdy_box.xc() - bdy_box.w()/2 + half_size + space_bdy
                                   : bdy_box.yc() - bdy_box.h()/2 + half_size + space_bdy;
            LocType bdy_Hi = isHor ? bdy_box.xc() + bdy_box.w()/2 - half_size - space_bdy
                                   : bdy_box.yc() + bdy_box.h()/2 - half_size - space_bdy;

            switch (chiplet_i->cover_reticles().size()) {
                case 1: { break; }
                case 2: { 
                    std::vector<Reticle*>& reticles = chiplet_i->cover_reticles(); 
                    if (!isHor && std::abs(reticles[0]->xc() - reticles[1]->xc()) < utils::epsilon) {
                        bdy_Lo = chiplet_i->yc()-1, bdy_Hi = chiplet_i->yc()+1;
                    } else if (isHor && std::abs(reticles[0]->yc() - reticles[1]->yc()) < utils::epsilon) {
                        bdy_Lo = chiplet_i->xc()-1, bdy_Hi = chiplet_i->xc()+1;
                    }
                    break;
                }
                case 4: {
                    bdy_Lo = isHor ? chiplet_i->xc()-1 : chiplet_i->yc()-1;
                    bdy_Hi = isHor ? chiplet_i->xc()+1 : chiplet_i->yc()+1;
                    break;
                }
                default: { spdlog::warn("[legalization] confusing reticle number: {} (id: {})", chiplet_i->cover_reticles().size(), chiplet_i->id()); break; }
            }

            // pin cluster boundary inside the connected chiplet's reticle
            if (false) { // enable pin cluster reticle constraints
                if (chiplet_i->reticle_id() == 0) {
                    for (ClusterType* cluster : chiplet_i->vclusters()) {
                        if (cluster->vpins().empty()) { continue; }
                        Chiplet* chiplet_j = cluster->connCluster()->chiplet();
                        Reticle* reticle_j = _db.getReticlewithId(chiplet_j->reticle_id());
                        if (chiplet_j->reticle_id() == 0 || reticle_j == nullptr) {
                            spdlog::error("[legalization] no unique target reticle for chiplet {} cluster {} connected to chiplet {}", chiplet_i->id(), cluster->id(), chiplet_j->id());
                            obj.end(); return false;
                        }

                        IntType rot_idx = chiplet_i->discrete_theta_idx();
                        IntType flip_idx = chiplet_i->discrete_theta_f_idx();
                        LocType cluster_Lo = LOC_TYPE_MAX, cluster_Hi = -LOC_TYPE_MAX;
                        for (Pin* pin : cluster->vpins()) {
                            Point offset = pin->point();
                            if (flip_idx == 1) { offset.x = -offset.x; }
                            for (IntType rot = 0; rot < rot_idx; rot++) { offset = Point(-offset.y, offset.x); }
                            LocType pin_w = rot_idx % 2 == 0 ? pin->w() : pin->h();
                            LocType pin_h = rot_idx % 2 == 0 ? pin->h() : pin->w();
                            LocType center = isHor ? offset.x : offset.y;
                            LocType half_pin = (isHor ? pin_w : pin_h) / 2.0;
                            cluster_Lo = std::min(cluster_Lo, center-half_pin);
                            cluster_Hi = std::max(cluster_Hi, center+half_pin);
                        }

                        LocType reticle_center = isHor ? reticle_j->xc() : reticle_j->yc();
                        LocType reticle_half = (isHor ? reticle_j->w() : reticle_j->h()) / 2.0;
                        bdy_Lo = std::max(bdy_Lo, reticle_center-reticle_half-cluster_Lo);
                        bdy_Hi = std::min(bdy_Hi, reticle_center+reticle_half-cluster_Hi);
                        if (bdy_Lo > bdy_Hi) {
                            spdlog::error("[legalization] infeasible {} bounds [{}, {}] for chiplet {} cluster {} in reticle {}", isHor ? "x" : "y", bdy_Lo, bdy_Hi, chiplet_i->id(), cluster->id(), reticle_j->id());
                            obj.end(); return false;
                        }
                    }
                }
            }

            var_i.setBounds(bdy_Lo, bdy_Hi);

        }

        // overlap
        ConsEdges graph_edges = generateConsEdges(isHor);
        // for (auto& edge : graph_edges.edges()) {
        //     std::cout << edge.toStr() << std::endl;
        // }
        for (auto& edge : graph_edges.edges()) { 
            IndexType chiplet_idx0 = edge.source();
            IndexType chiplet_idx1 = edge.target();

            if (chiplet_idx0 == _db.chiplets().size() || chiplet_idx1 == _db.chiplets().size()+1) { continue; }

            IloNumVar var_0 = solver.var(phys2var(chiplet_idx0, horver));
            IloNumVar var_1 = solver.var(phys2var(chiplet_idx1, horver));
            Chiplet* chiplet_0 = _db.chiplets()[chiplet_idx0].get();
            Chiplet* chiplet_1 = _db.chiplets()[chiplet_idx1].get();
            LocType  size_0 = isHor ? chiplet_0->rot_w() : chiplet_0->rot_h();
            LocType  size_1 = isHor ? chiplet_1->rot_w() : chiplet_1->rot_h();
            LocType space_0 = isHor ? chiplet_0->space()[1] : chiplet_0->space()[3];
            LocType space_1 = isHor ? chiplet_1->space()[0] : chiplet_1->space()[2];
            
            solver.LPaddConstraint(var_1-var_0, ConType::GreaterEq, size_0/2+size_1/2+space_0+space_1);
        }

        // objecitve
        for (auto& ptr_cpair : _db.cpairs()) {
            CPairType* cpair = ptr_cpair.get();
            Chiplet* chiplet_0 = cpair->chiplet0();
            Chiplet* chiplet_1 = cpair->chiplet1();

            IloNumVar var_0 = solver.var(phys2var(chiplet_0->id(), horver));
            IloNumVar var_1 = solver.var(phys2var(chiplet_1->id(), horver));

            Point offset_center_0 = cpair->cluster0()->rot_offset_center();
            Point offset_center_1 = cpair->cluster1()->rot_offset_center();
            LocType   offset_0 = isHor ? offset_center_0.x : offset_center_0.y;
            LocType   offset_1 = isHor ? offset_center_1.x : offset_center_1.y;

            IloExpr expr_0(solver.env()), expr_1(solver.env());
            expr_0 += (var_0 + offset_0);
            expr_1 += (var_1 + offset_1);
            IloNumVar d = solver.addAbsConstraint(expr_0-expr_1); 
            obj += static_cast<IloNum>(cpair->pins0().size()) * d;
            expr_0.end(); expr_1.end();
        }
    }

    solver.setMinimizeObjective(obj);

    // std::vector<RealType> init;
    // for (auto& ptr_chiplet : _db.chiplets()) {
    //     init.emplace_back(ptr_chiplet.get()->xc());
    //     init.emplace_back(ptr_chiplet.get()->yc());
    // }
    // solver.setInitialSolution(init);

    if (!solver.solve()) {
        spdlog::error("[legalization] No solution found!");
        return false;
    }

    if (direction == 0) {
        for (IndexType idx=0; idx<_db.chiplets().size(); idx++) {
            _db.chiplets()[idx].get()->xc() = solver.getVariableValue(idx);
        }
    } else if (direction == 1) {
        for (IndexType idx=0; idx<_db.chiplets().size(); idx++) {
            _db.chiplets()[idx].get()->yc() = solver.getVariableValue(idx);
        }
    } else {
        for (IndexType idx=0; idx<_db.chiplets().size(); idx++) {
            _db.chiplets()[idx].get()->xc() = solver.getVariableValue(2*idx);
            _db.chiplets()[idx].get()->yc() = solver.getVariableValue(2*idx+1);
        }
    }



    obj.end();
    spdlog::info("[legalization] finish");

    return true;
}
