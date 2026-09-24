#ifndef _ROUTABILITY_H
#define _ROUTABILITY_H

#include<queue>
#include<optional>
#include<string>

#include"type.h"
#include"utils.h"
#include"db.h"

namespace rb {

using namespace db;

enum class Direction { X, Y };



class Grid : public Box {
public:
    Grid(IndexType idx) : Box(idx) {}

    IntType&  cid() { return _c_id; }
    RealType& cap_x() { return _cap_x; }
    RealType& cap_y() { return _cap_y; }
    RealType& dem_x() { return _dem_x; }
    RealType& dem_y() { return _dem_y; }
    RealType& dem_diff_x() { return _dem_diff_x; }
    RealType& dem_diff_y() { return _dem_diff_y; }

    void clear_demand() { _dem_x = 0; _dem_y = 0; _dem_diff_x = 0; _dem_diff_y = 0; }

    IndexType getCapacitance(Direction dir) const {
        switch (dir) {
            case Direction::X: return _cap_x;
            case Direction::Y: return _cap_y;
            default: return 0;
        }
    }

    IndexType getDemandDiff(Direction dir) const {
        switch (dir) {
            case Direction::X: return _dem_diff_x;
            case Direction::Y: return _dem_diff_y;
            default: return 0;
        }
    }

private:
    IntType  _c_id = -1;
    RealType _cap_x;
    RealType _cap_y;
    RealType _dem_x = 0;
    RealType _dem_y = 0;
    RealType _dem_diff_x = 0;
    RealType _dem_diff_y = 0;
};


class GridManeger {
public:
    GridManeger(LocType size_gx, LocType size_gy, LocType d_linex, LocType d_liney, IndexType num_layer);
    
    Grid* getGridwithId(IndexType idx_x, IndexType idx_y) { 
        if (idx_x < 0 || idx_x >= _num_gx || idx_y < 0 || idx_y >= _num_gy) { return nullptr; }
        return _vgrids[idx_y*_num_gx+idx_x].get();
    }

    IndexType getNumGx() const { return _num_gx; }
    IndexType getNumGy() const { return _num_gy; }
    LocType getSizeGx() const { return _size_gx; }
    LocType getSizeGy() const { return _size_gy; }
    LocType d_linex() const { return _d_linex; }
    LocType d_liney() const { return _d_liney; }

    void updateCapacitance(Database& db);
    void clearDemand() { for (auto& ptr_grid : _vgrids) { ptr_grid.get()->clear_demand(); } }
    void clearCapacitance() { 
        for (auto& ptr_grid : _vgrids) { 
            ptr_grid.get()->cid() = -1;
            ptr_grid.get()->cap_x() = _num_layer * _size_gx / _d_linex;
            ptr_grid.get()->cap_y() = _num_layer * _size_gy / _d_liney;
        } 
    }

    void updateDiffuseDemand(IndexType range);
    
    void outputHotspot(NameType casepath, RealType threshold, IndexType num_call);
    std::vector<Box> extractHotspotBoxes(Direction line_direction, RealType threshold);

    void outputGrid(NameType casepath, bool en_diff, IndexType num_call);



private:
    RealType  _num_layer;
    IndexType _num_gx;
    IndexType _num_gy;
    LocType  _size_gx;
    LocType  _size_gy;
    LocType  _d_linex;
    LocType  _d_liney;
    std::vector<std::unique_ptr<Grid>> _vgrids;
};


class GridRouter {
public:
    GridRouter(std::string workdir, std::string casedir, Database& db, GridManeger& gm) : _workdir(workdir), _casedir(casedir), _db(db), _gm(gm) {}

    GridManeger& gridManeger() { return _gm; }

    bool updateChipletSpace();

    void routeAllCpair();
    
    std::vector<Path> routeFromNearestAvlGrid(CPairType* cpair, IndexType max_paths = 1000);

    void updateDemandFromPath(const Path& path, RealType dem);



    
    
    
private:
    std::string _workdir;
    std::string _casedir;
    Database& _db;
    GridManeger& _gm;

    RealType _hotspot_threshold = 0.8;
};






}







#endif

