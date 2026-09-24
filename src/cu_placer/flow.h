#ifndef FLOW_H_
#define FLOW_H_

#include<iostream>
#include<string>

#include"spdlog/spdlog.h"

#include"utils.h"
#include"db.h"
#include"gplacer.h"
#include"CGLegalizer.h"
#include"routability.h"

using namespace db;

class MainFlow {
    public: 
        MainFlow(std::string workdir, std::string casedir) : _workdir(workdir), _casedir(casedir) {

            spdlog::info("Hello Placer.");
            spdlog::info("    __  __________    __    ____     ____  __    ___   ________________ ");
            spdlog::info("   / / / / ____/ /   / /   / __ \\   / __ \\/ /   /   | / ____/ ____/ __ \\");
            spdlog::info("  / /_/ / __/ / /   / /   / / / /  / /_/ / /   / /| |/ /   / __/ / /_/ /");
            spdlog::info(" / __  / /___/ /___/ /___/ /_/ /  / ____/ /___/ ___ / /___/ /___/ _  _/ ");
            spdlog::info("/_/ /_/_____/_____/_____/\\____/  /_/   /_____/_/  |_\\____/_____/_/ |_|  ");

            _json_netlist      = utils::ReadJSON(_workdir+_casedir+"json_chiplets/chiplet.json");
            _json_interposer   = utils::ReadJSON(_workdir+_casedir+"json_chiplets/interposer.json");
            _json_datadescript = utils::ReadJSON(_workdir+_casedir+"place_init/case_discription.json");
            _v_reticle_msg     = utils::readIntTXT(_workdir+_casedir+"hur_partition/reticle_msg.txt");
            _v_reticle_adj     = utils::readIntTXT(_workdir+_casedir+"hur_partition/reticle_msg.txt");
            _v_partition_msg   = utils::readIntTXT(_workdir+_casedir+"hur_partition/chiplet_allocation.txt");
            _json_gpopt        = utils::ReadJSON(_workdir+_casedir+"strategy.json");

            utils::XSCALE = _v_reticle_msg(0, 2) / 100;
            utils::YSCALE = _v_reticle_msg(0, 3) / 100;
            utils::XSCALE = std::min(utils::XSCALE, utils::YSCALE);
            utils::YSCALE = utils::XSCALE;
            utils::WHITESPACE = _json_gpopt["whitespace"].asDouble();
            utils::EN_GPU = _json_gpopt["en_gpu"].asBool();
            
            spdlog::info("[mainflow] scale the size to 0~100 with XSCALE={}, YSCALE={}, chiplet whitespace={}", utils::XSCALE, utils::YSCALE, utils::WHITESPACE);
            LocType xmin = LOC_TYPE_MAX, ymin = LOC_TYPE_MAX;
            LocType xmax = LOC_TYPE_MIN, ymax = LOC_TYPE_MIN;
            for (IndexType i=0; i<_v_reticle_msg.n_rows; i++) { 
                xmin = std::min(xmin, static_cast<LocType>(_v_reticle_msg(i, 0)));
                ymin = std::min(ymin, static_cast<LocType>(_v_reticle_msg(i, 1)));
                xmax = std::max(xmax, static_cast<LocType>(_v_reticle_msg(i, 0)+_v_reticle_msg(i, 2)));
                ymax = std::max(ymax, static_cast<LocType>(_v_reticle_msg(i, 1)+_v_reticle_msg(i, 3)));
            }
            utils::TOTAL_BDY[0] = xmin, utils::TOTAL_BDY[1] = ymin, utils::TOTAL_BDY[2] = xmax, utils::TOTAL_BDY[3] = ymax;
            spdlog::info("[mainflow] the total boundary of the design is: xl={}, yl={}, xh={}, yh={}, and scale to {}, {}, {}, {}, respectively", \
                         utils::TOTAL_BDY[0], utils::TOTAL_BDY[1], utils::TOTAL_BDY[2], utils::TOTAL_BDY[3], \
                         utils::TOTAL_BDY[0]/utils::XSCALE, utils::TOTAL_BDY[1]/utils::YSCALE, utils::TOTAL_BDY[2]/utils::XSCALE, utils::TOTAL_BDY[3]/utils::YSCALE);
        }

        void DatabaseInit();
        void OutputNewLocation();
        

        void runflow() {
            DatabaseInit();
            gp::GPlacer gplacer(_workdir, _casedir, _db, _json_gpopt);
            // gplacer.dbgtest_value_grad();
            arma::vec x_out(_db.chiplets().size()*(2+utils::NUM_DISCRETE_VAR));

            gplacer.ParametersInit(true, true, true, true);
            gplacer.AdamOptimize(x_out, 0);

            rb::GridManeger gridmaneger(1, 1, 0.0333, 0.04, 2);
            rb::GridRouter gridrouter(_workdir, _casedir, _db, gridmaneger);
            gridrouter.routeAllCpair();
            
            Legalizer legalizer(_db);
            // legalizer.legalize(2);
            legalizer.legalize(1); legalizer.legalize(0);
            gplacer.print_geometric_msg();

            // if ((!utils::EN_GPU) && gplacer.CustomizationChiplet(_v_reticle_adj)) {
            //     x_out.set_size(_db.chiplets().size()*(2+utils::NUM_DISCRETE_VAR));
            //     gplacer.ParametersInit(false, true, true, false);
            //     gplacer.AdamOptimize(x_out, 1);
            //     // legalizer.legalize(1); legalizer.legalize(0);
            //     gplacer.print_geometric_msg();
            // }

            // routability
            bool congest_success = false;
            utils::TimerCounter num_leg_max(5+1);
            
            // while(!num_leg_max.tick()) {
            //     gridrouter.routeAllCpair();
            //     if (gridrouter.updateChipletSpace()) {
            //         legalizer.legalize(2);
            //         gplacer.print_geometric_msg();
            //     } else {
            //         break;
            //     }
            // }

            spdlog::info("write placement result");
            OutputNewLocation();
        }


    private:
        std::string _workdir;
        std::string _casedir;
        Json::Value _json_netlist;
        Json::Value _json_interposer;
        Json::Value _json_datadescript;
        LocMatrix _v_reticle_msg;
        LocMatrix _v_reticle_adj;
        LocMatrix _v_partition_msg;
        Json::Value _json_gpopt;

        db::Database _db;
        
};




#endif

