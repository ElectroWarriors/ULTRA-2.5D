#include"type.h"
#include"utils.h"
#include"db.h"

namespace db {
// 

IndexType ClusterType::_next_id = 0;
IndexType CPairType::_next_id   = 0;

bool segmentsIntersect(const Point& a, const Point& b, const Point& c, const Point& d) {
    auto cross    = [](const Point& p, const Point& q) { return p.x * q.y - p.y * q.x; };
    auto subtract = [](const Point& p, const Point& q) { return Point{p.x - q.x, p.y - q.y}; };

    Point ab = subtract(b, a), ac = subtract(c, a), ad = subtract(d, a);
    Point cd = subtract(d, c), ca = subtract(a, c), cb = subtract(b, c);

    RealType d0 = cross(ab, ac), d1 = cross(ab, ad), d2 = cross(cd, ca), d3 = cross(cd, cb);

    return d0 * d1 < 0 && d2 * d3 < 0;
}




void Chiplet::chiplet_init() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<RealType> dist(0.0, 1.0);
    for (auto& zi : _z) { zi = dist(gen); }
    _cached_gumbel_noise = utils::sample_gumbel<RealType, utils::NUM_DISCRETE_VAR>(utils::HPWL_PARAM1_GUMBEL);
    z2theta(utils::HPWL_PARAM0, utils::HPWL_PARAM1_GUMBEL);
}

void Chiplet::z2theta(RealType gamma, RealType gumbel) { 
    std::array<RealType, utils::NUM_DISCRETE_VAR> z_exp;
    if (utils::USE_GUMBEL) {
        z_exp = utils::gumbel_softmax_with_noise<RealType, utils::NUM_DISCRETE_VAR>(_z, gamma, _cached_gumbel_noise);
    } else {
        z_exp = utils::softmax<RealType, utils::NUM_DISCRETE_VAR>(_z, gamma);
    }
    // _z = z_exp;
    _theda = 0; _theda_f = 0;
    for (IndexType i=0; i<utils::NUM_DISCRETE_VAR; i++) { 
        _theda += z_exp[i] * utils::LEGAL_THETA[i]; 
        if (utils::USE_FLIP) {
            _theda_f += z_exp[i] * utils::LEGAL_THETA_F[i];
        }   
    }
    _cached_gumbel_noise = utils::sample_gumbel<RealType, utils::NUM_DISCRETE_VAR>(gumbel); // utils::HPWL_PARAM1_GUMBEL
}

void Chiplet::print_msg(NameType path) {
    std::string filename = path+"placer/log_"+utils::LOG_TIME+"/database_init/chiplet/chiplet_" + std::to_string(_id) + ".txt";
    auto logger = spdlog::basic_logger_mt("chiplet_logger", filename);

    logger->info("information for chiplet {}", _id);
    logger->info("location: ({0}x{1})", xc(), yc());
    logger->info("angle: ({})", _theda);
    logger->info("size: ({0}x{1})", w(), h());
    logger->info("pins number: ({0}x{1}->{2})", nx_pins(), ny_pins(), _vpins.size());
    logger->info("reticle id: ({})", _reticle_id);

    std::ostringstream oss;
    for (auto* r : _cover_reticles) { if (r) oss << r->id() << " "; }
    logger->info("cover reticle ids: {}", oss.str());

    for (auto* cl : _vclusters) { 
        logger->info("connect to chiplet {0} with {1} pins(={2})", cl->connCluster()->chiplet()->id(), cl->vpins().size(), cl->connCluster()->vpins().size());
    }

    logger->flush();
    spdlog::drop("chiplet_logger");
}

void Reticle::print_msg(NameType path) {
    std::string filename = path+"placer/log_"+utils::LOG_TIME+"/database_init/reticle/reticle_" + std::to_string(_id) + ".txt";
    auto logger = spdlog::basic_logger_mt("reticle_logger", filename);

    logger->info("information for reticle {}", _id);
    logger->info("location: ({0}x{1})", xc(), yc());
    logger->info("size: ({0}x{1})", w(), h());
    
    std::ostringstream oss;
    for (auto* c : _vchiplets) { if (c) oss << c->id() << " "; }
    logger->info("cotain chiplet ids: {}", oss.str());

    logger->flush();
    spdlog::drop("reticle_logger");
}



bool CPairType::addPinPair(Pin* pin0, Pin* pin1) {
    if (pin0->owner_chiplet()->id() == _chiplet0->id() && pin1->owner_chiplet()->id() == _chiplet1->id()) {
        _vpins0.emplace_back(pin0); _vpins1.emplace_back(pin1);
        return true;
    } else if (pin0->owner_chiplet()->id() == _chiplet1->id() && pin1->owner_chiplet()->id() == _chiplet0->id()) {
        _vpins0.emplace_back(pin1); _vpins1.emplace_back(pin0);
        return true;
    } else {
        return false;
    }
}

void CPairType::print_msg(NameType path) {
    std::string filename = path+"placer/log_"+utils::LOG_TIME+"/database_init/cpair/cpair_" + std::to_string(_id) + ".txt";
    auto logger = spdlog::basic_logger_mt("cpair_logger", filename);

    logger->info("connection between chiplet {0} and {1}, with {2}(={3}) nets", _chiplet0->id(), _chiplet1->id(), _vpins0.size(), _vpins1.size());
    logger->info("managed by cluster {} and {}", _cluster0->id(), _cluster0->id());

    logger->flush();
    spdlog::drop("cpair_logger");
}


void Database::findallChipletCoverReticles(IndexType id) {
    std::vector<Chiplet*> targets;

    if (id != INDEX_TYPE_MAX) {
        targets.emplace_back(getChipletwithId(id)); // 返回的是 Chiplet*
    } else {
        for (auto& ptr_chiplets : _vchiplets) {
            targets.emplace_back(ptr_chiplets.get()); // 提取裸指针
        }
    }

    for (auto ci : targets) {
        // Chiplet* ci = ptr_ci.get();
        ci->cover_reticles().clear();
        if (ci->reticle_id() != 0) { 
            Reticle* r = getReticlewithId(ci->reticle_id());
            // spdlog::info("{}", ci->reticle_id());
            ci->cover_reticles().emplace_back(r); 
            ci->boundaryBox().setLoc(r->xc(), r->yc());
            ci->boundaryBox().setSize(r->w(), r->h());
            continue; 
        } else {
            LocType xmin = LOC_TYPE_MAX, ymin = LOC_TYPE_MAX;
            LocType xmax = LOC_TYPE_MIN, ymax = LOC_TYPE_MIN;
            for (auto cluster : ci->vclusters()) {
                Reticle* rj = getReticlewithId(cluster->connCluster()->chiplet()->reticle_id());
                xmin = std::min(xmin, rj->xc()-rj->w()/2);
                xmax = std::max(xmax, rj->xc()+rj->w()/2);
                ymin = std::min(ymin, rj->yc()-rj->h()/2);
                ymax = std::max(ymax, rj->yc()+rj->h()/2);
            }
            Box box = Box((xmin+xmax)/2, (ymin+ymax)/2, xmax-xmin, ymax-ymin);
            for (auto& ptr_reticle : _vreticles) {
                Reticle* reticle = ptr_reticle.get();
                if (reticle->id() == 0) { continue; }
                Point ovl_xy = overlapXY<Point>(reticle, &box);
                if (ovl_xy.first() < 0 && ovl_xy.second() < 0) {
                    ci->cover_reticles().emplace_back(reticle);
                }
            }
            ci->boundaryBox().setLoc((xmin+xmax)/2, (ymin+ymax)/2);
            ci->boundaryBox().setSize(ci->cover_reticles()[0]->w(), ci->cover_reticles()[0]->h());
        }
    }
}


} // end namespace