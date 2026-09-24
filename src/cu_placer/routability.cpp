#include"type.h"
#include"utils.h"
#include"db.h"
#include"routability.h"

#include <algorithm>

namespace rb {

using namespace db;


GridManeger::GridManeger(LocType size_gx, LocType size_gy, LocType d_linex, LocType d_liney, IndexType num_layer) { // all have been scaled
    _num_gx  = (utils::TOTAL_BDY[2] - utils::TOTAL_BDY[0]) / utils::XSCALE / size_gx;
    _num_gy  = (utils::TOTAL_BDY[3] - utils::TOTAL_BDY[1]) / utils::YSCALE / size_gy;
    _size_gx = (utils::TOTAL_BDY[2] - utils::TOTAL_BDY[0]) / utils::XSCALE / _num_gx;
    _size_gy = (utils::TOTAL_BDY[3] - utils::TOTAL_BDY[1]) / utils::YSCALE / _num_gy;
    _d_linex = d_linex;
    _d_liney = d_liney;
    _num_layer = num_layer;

    for (IndexType idx_y = 0; idx_y < _num_gy; idx_y++) {
        for (IndexType idx_x = 0; idx_x < _num_gx; idx_x++) {
            Grid* grid = _vgrids.emplace_back(std::make_unique<Grid>(idx_y*_num_gx+idx_x)).get();
            grid->setLoc((1.0*idx_x+0.5)*_size_gx, (1.0*idx_y+0.5)*_size_gy);
            grid->setSize(_size_gx, _size_gy);
            grid->cap_x() = _num_layer * _size_gx / _d_linex;
            grid->cap_y() = _num_layer * _size_gy / _d_liney;
        }
    }
}


void GridManeger::updateCapacitance(Database& db) {
    clearCapacitance();
    clearDemand();

    for (auto& ptr_chiplet_i : db.chiplets()) {
        Chiplet* chipplet_i = ptr_chiplet_i.get();
        IndexType idx_xl = static_cast<IndexType>((chipplet_i->xc()-chipplet_i->rot_w()/2) / _size_gx);
        IndexType idx_xh = static_cast<IndexType>((chipplet_i->xc()+chipplet_i->rot_w()/2) / _size_gx) + 1;
        IndexType idx_yl = static_cast<IndexType>((chipplet_i->yc()-chipplet_i->rot_h()/2) / _size_gy);
        IndexType idx_yh = static_cast<IndexType>((chipplet_i->yc()+chipplet_i->rot_h()/2) / _size_gy) + 1;

        for (IndexType idx_y = idx_yl; idx_y <= idx_yh; idx_y++) {
            for (IndexType idx_x = idx_xl; idx_x <= idx_xh; idx_x++) {
                Grid* grid = getGridwithId(idx_x, idx_y); 
                if (!grid) { spdlog::warn("legalization out of range: (idx, idy) = ({}, {})", idx_x, idx_y); continue; }
                grid->cid()   = chipplet_i->id();
                // grid->cap_x() = _size_gx / _d_linex;
                // grid->cap_y() = _size_gy / _d_liney;
                
                if (idx_x == idx_xl || idx_x == idx_xh || idx_y == idx_yl || idx_y == idx_yh) {
                    Point ovl_xy = db::overlapXY<Point>(grid, chipplet_i);
                    if (ovl_xy.first() >= 0 && ovl_xy.second() >= 0) {
                        grid->cap_x() = _num_layer * std::max(0.0, (_size_gx - std::abs(ovl_xy.first() )) / _d_linex);
                        grid->cap_y() = _num_layer * std::max(0.0, (_size_gy - std::abs(ovl_xy.second())) / _d_liney);
                    }
                } else {
                    grid->cap_x() = 0; grid->cap_y() = 0;
                }
            }
        }
    }
}


void GridManeger::updateDiffuseDemand(IndexType range) {
    for (IndexType idx_y = 0; idx_y < _num_gy; idx_y++) {
        for (IndexType idx_x = 0; idx_x < _num_gx; idx_x++) {
            Grid* grid_c = getGridwithId(idx_x, idx_y);

            if (grid_c->dem_x() != 0) {
                std::vector<Grid*> valid_grids;
                for (IntType i = idx_y - range; i <= idx_y + range; i++) {
                    Grid* g = getGridwithId(idx_x, i);
                    if (g && g->cap_x() != 0) { valid_grids.push_back(g); }
                }

                if (!valid_grids.empty()) {
                    RealType share = grid_c->dem_x() / valid_grids.size();
                    for (Grid* g : valid_grids) { g->dem_diff_x() += share; }
                }
            }
            if (grid_c->dem_y() != 0) {
                std::vector<Grid*> valid_grids;
                for (IntType i = idx_x - range; i <= idx_x + range; i++) {
                    Grid* g = getGridwithId(i, idx_y);
                    if (g && g->cap_y() != 0) { valid_grids.push_back(g); }
                }

                if (!valid_grids.empty()) {
                    RealType share = grid_c->dem_y() / valid_grids.size();
                    for (Grid* g : valid_grids) { g->dem_diff_y() += share; }
                }
            }
        }
    }
}


void GridManeger::outputHotspot(NameType casepath, RealType threshold, IndexType num_call) {
    NameType basepath = casepath+"placer/log_"+utils::LOG_TIME+"/routability/";
    auto hotspot_logger = spdlog::basic_logger_mt("hotspot_logger", basepath+"hotspot_"+std::to_string(num_call++)+".txt");
    hotspot_logger->set_pattern("%v");

    for (IndexType i = 0; i < 2; i++) {
        Direction line_direction = (i == 0) ? Direction::X : Direction::Y;
        std::vector<Box> hotspots = extractHotspotBoxes(line_direction, threshold);
        NameType msg = (i == 0) ? "Horizontal" : "Vertical";
            std::ostringstream oss;
        for (Box& hotspot : hotspots) { 
            oss << "(" << hotspot.xc() << "," << hotspot.yc() << "," << hotspot.w() << "," << hotspot.h() << ")"; 
        }
        hotspot_logger->info("{} lines routability hotspot: {}", msg, oss.str()); 
    }
    hotspot_logger->flush();
    spdlog::drop("hotspot_logger");
}


std::vector<Box> GridManeger::extractHotspotBoxes(Direction line_direction, RealType threshold) {
    const IndexType rows = _num_gy;
    const IndexType cols = _num_gx;

    std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));
    std::vector<Box> boxes;
    // 8 neighbor
    const std::vector<std::pair<IntType, IntType>> directions = {
        {-1, -1}, {-1, 0}, {-1, 1},
        { 0, -1},          { 0, 1},
        { 1, -1}, { 1, 0}, { 1, 1}
    };
    // congestion ratio
    auto getRatio = [&](Grid* g) -> RealType {
        if (!g) return -1;
        RealType cap = g->getCapacitance(line_direction);
        if (cap <= 0) { return -1; }
        RealType dem = g->getDemandDiff(line_direction);
        return dem / cap;
    };

    // sweep each unvisited grid cell
    for (IntType r = 0; r < rows; ++r) { // for each y
        for (IntType c = 0; c < cols; ++c) { // for each x
            Grid* g = getGridwithId(c, r);
            RealType ratio = getRatio(g);

            if (ratio > threshold && !visited[r][c]) {
                IntType r_min = r, r_max = r;
                IntType c_min = c, c_max = c;

                std::queue<std::pair<IntType, IntType>> q;
                q.push({r, c});
                visited[r][c] = true;

                // BFS 
                while (!q.empty()) {
                    auto [x, y] = q.front(); q.pop();

                    for (const auto& [dx, dy] : directions) {
                        IntType nx = x + dx;
                        IntType ny = y + dy;

                        Grid* ng = getGridwithId(ny, nx);
                        if (ng && !visited[nx][ny] && ng->getCapacitance(line_direction) > 0) {
                            RealType nr = getRatio(ng);
                            if (nr > threshold) {
                                visited[nx][ny] = true;
                                q.push({nx, ny});
                                r_min = std::min(r_min, nx);
                                r_max = std::max(r_max, nx);
                                c_min = std::min(c_min, ny);
                                c_max = std::max(c_max, ny);
                            }
                        }
                    }
                }

                // hotspot box 的几何中心与尺寸（物理单位）
                LocType xc = (c_min + c_max + 1) / 2.0 * getSizeGx();
                LocType yc = (r_min + r_max + 1) / 2.0 * getSizeGy();
                LocType w  = (c_max - c_min + 1) * getSizeGx();
                LocType h  = (r_max - r_min + 1) * getSizeGy();

                if(w < 1.1 &&  h < 1.1) { continue; }
                boxes.emplace_back(xc, yc, w, h);
            }
        }
    }

    return boxes;
}




void GridManeger::outputGrid(NameType casepath, bool en_diff, IndexType num_call) {
    NameType basepath = casepath+"placer/log_"+utils::LOG_TIME+"/routability/";
    auto grid_logger = spdlog::basic_logger_mt("grid_logger", basepath+"grid_"+std::to_string(num_call++)+".txt");
    grid_logger->set_pattern("%v");
    for (IndexType idx_y = 0; idx_y < _num_gy; idx_y++) {
        for (IndexType idx_x = 0; idx_x < _num_gx; idx_x++) {
            Grid* grid_i = getGridwithId(idx_x, idx_y);
            // #center_x, center_y, width, height, grid.cap_x(), grid.cap_y(), grid.dem_x(), grid.dem_y()
            if (!en_diff) {
                grid_logger->info("{} {} {} {} {} {} {} {}", grid_i->xc(), grid_i->yc(), grid_i->w(), grid_i->h(), \
                                                            grid_i->cap_x(), grid_i->cap_y(), grid_i->dem_x(), grid_i->dem_y());
            } else {
                grid_logger->info("{} {} {} {} {} {} {} {}", grid_i->xc(), grid_i->yc(), grid_i->w(), grid_i->h(), \
                                                            grid_i->cap_x(), grid_i->cap_y(), grid_i->dem_diff_x(), grid_i->dem_diff_y());
            }
        }
    }
    grid_logger->flush();
    spdlog::drop("grid_logger");
}

// ------------------------------------------------------------------------------------------------------------- //

bool GridRouter::updateChipletSpace() {
    bool is_congest = false;
    for (IndexType i = 0; i < 2; i++) {
        Direction line_direction = (i == 0) ? Direction::X : Direction::Y;
        std::vector<Box> hotspots = _gm.extractHotspotBoxes(line_direction, _hotspot_threshold);
        if (hotspots.size() == 0) { continue; }
        is_congest = true;
        // 
        for (Box& hotspot : hotspots) {
            IndexType idx_l  = 0;
            IndexType idx_h  = (line_direction == Direction::X) ? _gm.getNumGx() : _gm.getNumGy();
            IndexType idx_cl = static_cast<IndexType>((line_direction == Direction::X) ? hotspot.yc() : hotspot.xc());
            IndexType idx_cc = static_cast<IndexType>((line_direction == Direction::X) ? hotspot.xc() : hotspot.yc());
            IndexType num_grid = 0; RealType dem = 0;

            RealType size_grid = (line_direction == Direction::X) ? _gm.getSizeGx() : _gm.getSizeGy();
            RealType dist_line = (line_direction == Direction::X) ? _gm.d_linex()   : _gm.d_liney();
            RealType cap_per_grid = size_grid / dist_line;
            IndexType chiplet_l_idx = -1, chiplet_h_idx = -1;
            for (IndexType i = idx_cl; i > idx_l; i--) {
                Grid* g = (line_direction == Direction::X) ? _gm.getGridwithId(idx_cc, i) : _gm.getGridwithId(i, idx_cc);
                if (!g) { spdlog::warn("updateChipletSpace out of range: {} (stay), {} (loop)", idx_cc, i); break; }
                if (g->cid() == -1) { RealType g_dem = (line_direction == Direction::X) ? g->dem_x() : g->dem_y(); dem += g_dem; num_grid++; }
                else { chiplet_l_idx = g->cid(); break; }
            }
            if (chiplet_l_idx == -1) { continue; }

            for (IndexType i = idx_cl; i < idx_h; i++) {
                Grid* g = (line_direction == Direction::X) ? _gm.getGridwithId(idx_cc, i) : _gm.getGridwithId(i, idx_cc);
                if (!g) { spdlog::warn("updateChipletSpace out of range: {} (stay), {} (loop)", idx_cc, i); break; }
                if (g->cid() == -1) { RealType g_dem = (line_direction == Direction::X) ? g->dem_x() : g->dem_y(); dem += g_dem; num_grid++; }
                else { chiplet_h_idx = g->cid(); break; }
            }
            if (chiplet_h_idx == -1) { continue; }

            if (dem < 0.9 * cap_per_grid * num_grid) { continue; }

            RealType need_space = (dem / (0.9 * cap_per_grid) + 1) * size_grid;

            Chiplet* chiplet_l = _db.chiplets()[chiplet_l_idx].get(); 
            Chiplet* chiplet_h = _db.chiplets()[chiplet_h_idx].get();
            LocType& chiplet_l_expand = (line_direction == Direction::X) ? chiplet_l->space()[3] : chiplet_l->space()[1];
            LocType& chiplet_h_expand = (line_direction == Direction::X) ? chiplet_h->space()[2] : chiplet_h->space()[0];
            chiplet_l_expand += need_space / 2;
            chiplet_h_expand += need_space / 2;
        }
    }

    return is_congest;
}


void GridRouter::routeAllCpair() {
    static IndexType num_call = 0;

    _gm.updateCapacitance(_db);

    spdlog::info("[routability] start");
    NameType basepath = _workdir+_casedir+"placer/log_"+utils::LOG_TIME+"/routability/";
    auto path_logger = spdlog::basic_logger_mt("path_logger", basepath+"path_"+std::to_string(num_call)+".txt");
    path_logger->set_pattern("%v");  

    spdlog::info("[routability] generate path");
    for (auto& ptr_cpair : _db.cpairs()) {
        CPairType* cpair_i = ptr_cpair.get();
        std::vector<Path> paths = routeFromNearestAvlGrid(cpair_i);

        path_logger->info("# Paths: CPair {}", cpair_i->id());
        for (IndexType idx = 0; idx < paths.size(); idx++) {
            Path& path = paths[idx];

            std::ostringstream oss;
            for (CoordId& coord : path) { 
                Point p = _gm.getGridwithId(coord.first, coord.second)->point(); 
                oss << "(" << p.x << "," << p.y << ")"; 
            }
            path_logger->info("## Path {}: {}", idx, oss.str());
        }
    }
    path_logger->flush();
    spdlog::drop("path_logger");

    spdlog::info("[routability] diffuse");
    _gm.updateDiffuseDemand(10);
    spdlog::info("[routability] output");
    _gm.outputGrid(_workdir+_casedir, true, num_call);
    _gm.outputHotspot(_workdir+_casedir, _hotspot_threshold, num_call);
    num_call++;
}

std::vector<Path> GridRouter::routeFromNearestAvlGrid(CPairType* cpair, IndexType max_paths) {
    Point op = cpair->cluster0()->rot_offset_center(), tp = cpair->cluster1()->rot_offset_center();
    CoordId origin = {(cpair->chiplet0()->xc() + op.x) / _gm.getSizeGx(), 
                      (cpair->chiplet0()->yc() + op.y) / _gm.getSizeGy()};
    CoordId target = {(cpair->chiplet1()->xc() + tp.x) / _gm.getSizeGx(), 
                      (cpair->chiplet1()->yc() + tp.y) / _gm.getSizeGy()};
    const IndexType rows = _gm.getNumGy();
    const IndexType cols = _gm.getNumGx();
    const std::array<CoordId, 4> directions = {{ {-1, 0}, {1, 0}, {0, -1}, {0, 1} }};
    const std::array<Direction, 4> dir_enum = { Direction::X, Direction::X, Direction::Y, Direction::Y };

    // Step 1: BFS from origin to find origin and target nearest cap>0 point in given direction
    std::optional<CoordId> origin_nearest;
    std::optional<CoordId> target_nearest;
    // 
    std::vector<std::vector<bool>> visited(rows, std::vector<bool>(cols, false));
    std::queue<CoordId> queue;
    queue.push(origin);
    visited[origin.second][origin.first] = true;
    while (!queue.empty()) {
        auto [x, y] = queue.front();
        queue.pop();

        Grid* g = _gm.getGridwithId(x, y);
        if (g && (g->getCapacitance(Direction::X) > 0 || g->getCapacitance(Direction::Y) > 0)) {
            origin_nearest = {x, y};
            // std::cout << x << " " << y << std::endl;
            break;
        }

        for (auto [dx, dy] : directions) {
            IntType nx = x + dx;
            IntType ny = y + dy;
            if (nx >= 0 && nx < cols && ny >= 0 && ny < rows && !visited[ny][nx]) {
                visited[ny][nx] = true;
                queue.push({nx, ny});
            }
        }
    }
    if (!origin_nearest) return {};  // no accessable point
    // 
    queue = std::queue<CoordId>();
    queue.push(target);
    visited.assign(rows, std::vector<bool>(cols, false));
    visited[target.second][target.first] = true;
    while (!queue.empty()) {
        auto [x, y] = queue.front();
        queue.pop();

        Grid* g = _gm.getGridwithId(x, y);
        if (g && (g->getCapacitance(Direction::X) > 0 || g->getCapacitance(Direction::Y) > 0)) {
            target_nearest = {x, y};
            // std::cout << x << " " << y << std::endl;
            break;
        }

        for (auto [dx, dy] : directions) {
            IntType nx = x + dx;
            IntType ny = y + dy;
            if (nx >= 0 && nx < cols && ny >= 0 && ny < rows && !visited[ny][nx]) {
                visited[ny][nx] = true;
                queue.push({nx, ny});
            }
        }
    }
    if (!target_nearest) return {};  // no accessable point

    // Step 2: BFS from target to build distance map
    std::vector<std::vector<IndexType>> dist(rows, std::vector<IndexType>(cols, -1));
    std::queue<CoordId> bfs_queue;
    bfs_queue.push(*target_nearest);
    dist[target_nearest->second][target_nearest->first] = 0;

    while (!bfs_queue.empty()) {
        auto [x, y] = bfs_queue.front();
        bfs_queue.pop();
        for (IndexType i = 0; i < directions.size(); ++i) {
            auto [dx, dy] = directions[i];
            Direction d = dir_enum[i];
            IntType nx = x + dx;
            IntType ny = y + dy;
            if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) continue;
            if (dist[ny][nx] != -1) continue;
            Grid* g = _gm.getGridwithId(nx, ny);
            if (g && g->getCapacitance(d) > 0) {
                dist[ny][nx] = dist[y][x] + 1;
                bfs_queue.push({nx, ny});
            }
        }
    }

    if (dist[origin_nearest->second][origin_nearest->first] == -1) return {};  // 无路径

    // Step 3: DFS to enumerate all shortest paths
    std::vector<Path>& paths = cpair->v_paths();
    paths.clear();
    std::function<void(Path&)> dfs = [&](Path& path) {
        if (paths.size() >= max_paths) return;
        auto [x, y] = path.back();
        if (x == target_nearest->first && y == target_nearest->second) {
            paths.push_back(path);
            return;
        }
        for (IndexType i = 0; i < directions.size(); ++i) {
            auto [dx, dy] = directions[i];
            Direction d = dir_enum[i];
            IntType nx = x + dx;
            IntType ny = y + dy;
            if (nx < 0 || nx >= cols || ny < 0 || ny >= rows) continue;
            Grid* g = _gm.getGridwithId(nx, ny);
            if (g && g->getCapacitance(d) > 0 && dist[ny][nx] == dist[y][x] - 1) {
                path.emplace_back(nx, ny);
                dfs(path);
                path.pop_back();
            }
        }
    };

    Path initial = {*origin_nearest};
    dfs(initial);

    // Step 4: Update dem_x / dem_y for each path
    // spdlog::info("wirelength between chiplet {} and {}: {} u", cpair->chiplet0()->id(), cpair->chiplet1()->id(), paths[0].size()*utils::XSCALE);
    for (const auto& path : paths) { updateDemandFromPath(path, 1.0*cpair->pins0().size()/paths.size()); }

    return paths;
}


void GridRouter::updateDemandFromPath(const Path& path, RealType dem) {
    IndexType num_path = path.size();
    if (num_path < 2) return;

    for (IndexType i = 1; i < num_path; ++i) {
        const auto& from = path[i - 1];
        const auto& to   = path[i];

        IntType dx = to.first  - from.first, dy = to.second - from.second;

        Direction dir;
        if (dx != 0 && dy == 0) { dir = Direction::X; } 
        else if (dy != 0 && dx == 0) { dir = Direction::Y; } 
        else { continue;}  // skip invalid or diagonal moves

        Grid* g = _gm.getGridwithId(from.first, from.second);
        if (!g) continue;

        bool turned = false;
        if (i >= 2) {
            const auto& prev = path[i - 2];
            IntType pdx = from.first  - prev.first, pdy = from.second - prev.second;
            if ((pdx != dx) || (pdy != dy)) { turned = true; }
        }

        if (turned) { g->dem_x() += dem; g->dem_y() += dem; } 
        else {
            if (dir == Direction::X) { g->dem_x() += dem; } else { g->dem_y() += dem; }
        }

        if (i == num_path - 1) {
            g = _gm.getGridwithId(to.first, to.second);
            if (dir == Direction::X) { g->dem_x() += dem; } else { g->dem_y() += dem; }
        }
    }
}










}


