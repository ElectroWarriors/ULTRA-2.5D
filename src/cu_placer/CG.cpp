#include"CG.h"

void SweeplineCGGenerator::solve() {
    originalSweepLine(); // TCAD-87
}


void originalInsert(const CellCoord &cellCoord, SweeplineCGGenerator::CellCoordTree &dTree, std::vector<IntType> &cand) {
    dTree.insert(cellCoord);
    cand.at(cellCoord.cellIdx()) = dTree.left(cellCoord);
    cand.at(dTree.right(cellCoord)) = cellCoord.cellIdx(); // dTree.right should never be -1. Because R0 is always there
}

void SweeplineCGGenerator::originalDelete(const CellCoord &cellCoord, CellCoordTree &dTree, std::vector<IntType> &cand, ConsEdges &cs) {
    auto left = dTree.left(cellCoord);
    if (left != -1 && left == cand.at(cellCoord.cellIdx())) {
        IntType from = left;
        IntType to = cellCoord.cellIdx();
        IntType weight = 0; //< We actually don't care. In the original algorithm, the weight is the width of the left, and it uses shortest path to compact the layout.
        if (!isExempted(from, to)) {
            cs.addConstraintEdge(from, to, weight);
        }
    }
    auto right = dTree.right(cellCoord);
    if (cand.at(right) == static_cast<IntType>(cellCoord.cellIdx())) {
        IntType from = cellCoord.cellIdx();
        IntType to = right;
        IntType weight = 0;
        if (!isExempted(from, to)) {
            cs.addConstraintEdge(from, to, weight);
        }
    }
    dTree.erase(cellCoord);
}


void SweeplineCGGenerator::originalConstraintGeneration(ConsEdges &cs, std::vector<Event> &events, std::vector<CellCoord> &cellCoords) {
    SweeplineCGGenerator::CellCoordTree dTree;
    IndexType numCells = cellCoords.size();
    std::vector<IntType> cand(numCells + 2, -1);
    // Insert R0
    dTree.insert(CellCoord(numCells + 1, LOC_TYPE_MAX)); // R0 is behaving as the target node in the DAG. which in convention is numCells + 1.
    for (const auto &event : events) {
        if (event.isLow()) {
            originalInsert(cellCoords.at(event.cellIdx()), dTree, cand);
        } else {
            originalDelete(cellCoords.at(event.cellIdx()), dTree, cand, cs);
        }
    }
}

void SweeplineCGGenerator::recordCellCoords(std::vector<CellCoord> &cellCoords, bool isHor) {
    cellCoords.clear();
    for (IndexType cellIdx = 0; cellIdx < _db.chiplets().size(); ++cellIdx) {
        const auto& cell = _db.getChipletwithId(cellIdx);
        if (isHor) { cellCoords.emplace_back(CellCoord(cellIdx, cell->xc()-cell->rot_w()/2)); } 
        else       { cellCoords.emplace_back(CellCoord(cellIdx, cell->yc()-cell->rot_h()/2)); }
    }
}

void SweeplineCGGenerator::generateEvents(std::vector<Event> &events, bool isHor) {
    events.clear();
    for (IndexType cellIdx = 0; cellIdx < _db.chiplets().size(); ++cellIdx) {
        const auto &cell = _db.getChipletwithId(cellIdx);
        if (isHor) {
            events.emplace_back(Event(cellIdx, cell->yc()-cell->rot_h()/2, true ));
            events.emplace_back(Event(cellIdx, cell->yc()+cell->rot_h()/2, false));
        } else {
            events.emplace_back(Event(cellIdx, cell->xc()-cell->rot_w()/2, true ));
            events.emplace_back(Event(cellIdx, cell->xc()+cell->rot_w()/2, false));
        }
    }
    std::sort(events.begin(), events.end());
}

void addEdgesFromSource(ConsEdges &cs, IndexType numCells) {
    std::vector<char> hasIngoingEdges(numCells + 2, false);
    for (const auto &edge : cs.edges()) {
        IndexType to = edge.target();
        hasIngoingEdges.at(to) = true;
    }
    IndexType sourceIdx = numCells;
    for (IndexType idx = 0; idx < numCells; ++idx) {
        if (!hasIngoingEdges.at(idx)) {
            IntType from = sourceIdx;
            IntType to = idx;
            IntType weight = 0; // Don't care
            cs.addConstraintEdge(from, to, weight);
        }
    }
}

void SweeplineCGGenerator::originalSweepLine() {
    std::vector<Event> events;
    std::vector<CellCoord> cellCoords;
    // First finding horizontal edges
    generateEvents(events, true);
    recordCellCoords(cellCoords, true);
    originalConstraintGeneration(_hC, events, cellCoords);
    addEdgesFromSource(_hC, cellCoords.size());
    // Vertical
    generateEvents(events, false);
    recordCellCoords(cellCoords, false);
    originalConstraintGeneration(_vC, events, cellCoords);
    addEdgesFromSource(_vC, cellCoords.size());
}