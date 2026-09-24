#ifndef CG_LEGALIZER_H_
#define CG_LEGALIZER_H_

#include <iostream>

#include"type.h"
#include"db.h"
#include "CG.h"

#include "json/json.h"

class Legalizer {
private:
    class BoxNode {
    public:
        BoxNode(IndexType cellIdx_, Point loc_, Size size_) {
            cellIdx = cellIdx_;
            loc = loc_;
            size = size_;
        }
        LocType coord; ///< Coordinate of the edge
        IndexType cellIdx; ///< The index of the cell 
        Point loc;
        Size size;
        std::vector<IndexType> prevlst;
        std::vector<IndexType> nextlst;
    };
    

public:
    Legalizer(Database& db) : _db(db) {}
    ConsEdges generateConsEdges(bool isHorG);
    void generateHorConsEdges();
    void generateVerConsEdges();
    bool legalize(IndexType direction);


private:
    Database& _db; ///< The database of IdeaPlaceEx

    ConstraintGraph _hCG; ///< The horizontal constraint graph
    ConstraintGraph _vCG; ///< The vertical constraint graph
    ConsEdges _hConsEdges; ///< The horizontal constraint edges
    ConsEdges _vConsEdges; ///< The vertical constraint edges

    std::vector<BoxNode> _allnodes;
};

#endif
