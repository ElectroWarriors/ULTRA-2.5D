

#ifndef CG_H_
#define CG_H_

#include"type.h"
#include"db.h"

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>

using namespace db;


class ConstraintEdge {
    public:
        explicit ConstraintEdge(IndexType source, IndexType target): _source(source), _target(target) {}
        IndexType source() const { return _source; }
        IndexType target() const { return _target; }
        IntType weight() const { return 1; }
        std::string toStr() const {
            std::stringstream ss;
            ss << "source "<< _source <<" target "<< _target ;
            return ss.str();
        }
        bool operator<(const ConstraintEdge &rhs) const {
            if (_source == rhs.source()) { return _target < rhs.target(); }
            return _source < rhs.source();
        }
        bool operator==(const ConstraintEdge &rhs) const {
            return _source == rhs.source() && _target == rhs.target();
        }
    private:
        IndexType _source;  ///< The index of source vertex
        IndexType _target; ///< The index of target vertex
        //IntType _weight;  ///< The weight of this edge
};

class ConsEdges {
    public:
        explicit ConsEdges() = default;
        void clear() { _edges.clear(); }
        const std::set<ConstraintEdge>& edges() const { return _edges; }
        std::set<ConstraintEdge>& edges() { return _edges; }
        void addConstraintEdge(IndexType sourceIdx, IndexType targetIdx, IntType weight) {
            _edges.insert(ConstraintEdge(sourceIdx, targetIdx));
        }
        bool hasEdgeNoDirection(IndexType sourceIdx, IndexType targetIdx) {
            auto it = _edges.find(ConstraintEdge(sourceIdx, targetIdx));
            if (it != _edges.end()) {
                return true;
            }
            it = _edges.find(ConstraintEdge(targetIdx, sourceIdx));
            if (it != _edges.end()) {
                return true;
            }
            return false;
        }
        void removeConstraintEdge(IndexType sourceIdx, IndexType targetIdx) {
            auto it = _edges.find(ConstraintEdge(sourceIdx, targetIdx));
            if (it != _edges.end()) {
                _edges.erase(it);
            }
            it = _edges.find(ConstraintEdge(targetIdx, sourceIdx));
            if (it != _edges.end()) {
                _edges.erase(it);
            }
        }
        
    private:
        std::set<ConstraintEdge> _edges; ///< The constraint edges
};


class ConstraintGraph {
    public:
        /// @brief default constructor
        explicit ConstraintGraph() {
            _cg.clear();
        }
        typedef boost::adjacency_list < boost::setS, boost::vecS, boost::bidirectionalS, boost::no_property, boost::property < boost::edge_weight_t, IntType > > graph_t;
        typedef boost::graph_traits < graph_t >::vertex_descriptor vertex_descriptor;
        typedef boost::graph_traits < graph_t >::edge_descriptor edge_descriptor;
        typedef boost::graph_traits < graph_t >::edge_iterator edge_iterator;
        typedef boost::property_map<graph_t, boost::vertex_index_t>::type IndexMap;
        typedef boost::graph_traits < graph_t >::adjacency_iterator adjacency_iterator;

        /// @brief return the boost graph
        graph_t & boostGraph() { return _cg; }
        /// @brief construct the graph with number of vertices
        /// @param the number of vertices
        void allocateVertices(IndexType numVex) { _cg.clear(); _cg = graph_t(numVex);}
        /// @brief get the number of nodes
        /// @return the number of nodes
        IndexType numNodes() const { return boost::num_vertices(_cg); }
        /// @brief get the source node index
        /// @return the index of the source node
        IndexType sourceNodeIdx() const { return numNodes() - 2;}
        /// @brief get the target node index
        /// @return the index of the target node
        IndexType targetNodeIdx() const { return numNodes() - 1; }
        /// @brief get the number of cells 
        /// @return the number of cell nodes in the graph
        IndexType numCellNodes() const { return numNodes() - 2; }
        /// @brief add edge to the graph
        /// @param the source node index
        /// @param the target node index
        void addEdge(IndexType sourceIdx, IndexType targetIdx, IntType weight=1) {
            boost::add_edge(boost::vertex(sourceIdx, _cg),
                    boost::vertex(targetIdx, _cg),
                    weight, _cg);
        }
        /// @brief remove a edge from the graph
        /// @param the source index
        /// @param the target index
        void removeEdge(IndexType sourceIdx, IndexType targetIdx) {
            boost::remove_edge(boost::vertex(sourceIdx, _cg), 
                    boost::vertex(targetIdx, _cg), 
                    _cg);
        }
        /// @brief determine whether the graph has one specific edge
        /// @param the source index of the edge
        /// @param the target index of the edge
        /// @return true if has edge. false if not
        bool hasEdge(IndexType sourceIdx, IndexType targetIdx) {
            auto edge = boost::edge(boost::vertex(sourceIdx, _cg),
                    boost::vertex(targetIdx, _cg), _cg);
            return edge.second;
        }
        
        void clear() {
            _cg.clear();
        }

    private:
        graph_t _cg; ///< The boost graph
};


class Event {
    public:
        explicit Event(IndexType cellIdx, LocType loc, bool isLow) : _cellIdx(cellIdx), _loc(loc), _isLow(isLow) {}
        /// @brief get the cell index this event belonging to
        IndexType cellIdx() const { return _cellIdx; }
        /// @brief get the coordinate of this event
        LocType loc() const { return _loc; }
        /// @brief get whether this event is low
        /// @return true: low. false: high
        bool isLow() const { return _isLow; }
        /// @brief comparing operator. The events are sorted in increasing loc order. When two coordinates are the same, the high events must before the low events. NOTE: what if there are multiple same coodinates?
        bool operator<(const Event &rhs) const {
            if (_loc == rhs.loc()) {
                if (_isLow == rhs.isLow()) { return _cellIdx < rhs.cellIdx(); }
                return !_isLow; // if lhs is low, lhs > rhs. if lhs is high (!_isLow), lhs < rhs.
            }
            return _loc < rhs.loc();
        }

    private:
        IndexType _cellIdx = INDEX_TYPE_MAX; ///< The index of cell this event belonging to
        LocType _loc = LOC_TYPE_MIN; ///< The coordinate of the event. If finding horzional edges, the loc is the lower/higher y edge.
        bool _isLow = true; ///< true: low edge. false: high edge
};

/// @brief a simple data structure to contain the information of the xLo/yLo
class CellCoord {
    public:
        explicit CellCoord(IndexType cellIdx, LocType loc) : _cellIdx(cellIdx), _loc(loc) {}
        IndexType cellIdx() const { return _cellIdx; }
        LocType loc() const { return _loc; }
        /// @brief comparison operator. \lambda relation in the original paper x_i > x_j or (xi = xj and i > j). Here take itsopposite
        bool operator<(const CellCoord &rhs) const {
            if (_loc == rhs.loc()) { return _cellIdx < rhs.cellIdx(); }
            return _loc < rhs.loc();
        }

    private:
        IndexType _cellIdx; ///< The cell index
        LocType _loc; ///< xLo or yLo
};

/// @class Sweep line for generating ConsEdges
class SweeplineCGGenerator {
    public:
        explicit SweeplineCGGenerator(Database& db, ConsEdges &hC, ConsEdges &vC) : _db(db), _hC(hC), _vC(vC) {}
        /// @brief solve the sweep line
        void solve();
        void setExemptFunc(std::function<bool(IndexType, IndexType)> exexmptFunc) {
            _exemptFunc = exexmptFunc;
            _setExempted = true;
        }
    /// @brief the balance tree for containing the "D" in TCAD-1987. Using the std::set implementation
    class CellCoordTree {
        public:
            explicit CellCoordTree() = default;
            /// @brief insert a CellCoord
            void insert(const CellCoord &cellCoord) { _tree.insert(cellCoord); }
            /// @brief erase a CellCoord
            void erase(const CellCoord &cellCoord) { _tree.erase(cellCoord); }
            /// @brief find the right cell index in the tree
            /// @param a CellCoord, this object should already inside the tree
            /// @return the cell index. -1 if nil
            IntType right(const CellCoord &cellCoord)
            {
                auto findIter = _tree.find(cellCoord);
                // Assert(findIter != _tree.end());
                if (findIter == _tree.end()) { throw std::runtime_error("findIter reached end of _tree"); }
                findIter++;
                if (findIter == _tree.end()) { return -1; }
                else { return static_cast<IntType>(findIter->cellIdx()); }
            }
            /// @brief find the left cell index in the tree
            /// @param a CellCoord, this object should already inside the tree
            /// @return the cell index. -1 if nil
            IntType left(const CellCoord &cellCoord)
            {
                auto findIter = _tree.find(cellCoord);
                // Assert(findIter != _tree.end());
                if (findIter == _tree.end()) { throw std::runtime_error("findIter reached end of _tree"); }
                if (findIter == _tree.begin())
                {
                    return -1; // This cellCoord is the leftmost in the tree
                }
                else
                {
                    findIter --;
                    return static_cast<IntType>(findIter->cellIdx());
                }
            }
        private:
            std::set<CellCoord> _tree; ///< std::set to act as the balanced tree
    };
    private:
        /// @brief generate the events.
        /// @param first: the referece to a vector of events. The vector will be cleared first.
        /// @param second: true: generating horizontal edges. false: generating vertical edges
        void generateEvents(std::vector<Event> &events, bool isHor);
        /// @brief original TCAD-87 algorithm
        void originalSweepLine();
        /// @brief record all the cell coordinates
        /// @param first: A reference to the vector of cell coordinates. The vector will be cleared in the beginning
        /// @param second: true: if generating horizontal edges. false if vertical
        void recordCellCoords(std::vector<CellCoord> &cellCords, bool isHor);
        /// @brief generate the constraint edges with TCAD-1987 compact ver. 1
        /// @param first: the constraints to save results in
        /// @param second: the sorted events
        /// @param third: the recorded cell coordinates
        //void originalConstraintGeneration(ConsEdges &cs, std::vector<Event> &events, std::vector<CellCoord> &cellCoords);
        void originalDelete(const CellCoord &cellCoord, CellCoordTree &dTree, std::vector<IntType> &cand, ConsEdges &cs);
        void originalConstraintGeneration(ConsEdges &cs, std::vector<Event> &events, std::vector<CellCoord> &cellCoords);
        bool isExempted(IndexType cell1, IndexType cell2) const
        {
            if (!_setExempted)
            {
                return false;
            }
            else
            {
                return _exemptFunc(cell1, cell2);
            }
        }

    private:
        Database &_db; ///< The placement database
        ConsEdges &_hC; ///< The horizontal edges
        ConsEdges &_vC; ///< The vertical edges
        std::function<bool(IndexType, IndexType)> _exemptFunc; ///< exempt pair of cells to be add constraints
        bool _setExempted = false;
};




#endif


