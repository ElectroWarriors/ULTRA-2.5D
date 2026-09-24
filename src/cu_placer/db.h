#ifndef DB_H_
#define DB_H_

#include<iostream>
#include <sstream>
#include<cstdint>  // UINT32_MAX etc.
#include<vector>
#include<string>
#include<cmath>
#include <random>
#include <array>
#include <algorithm>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>

#include"type.h"
#include"utils.h"
// #include"routability.h"




// template <typename T, typename U>
// std::pair<T, U> operator+(const std::pair<T, U>& lhs, const std::pair<T, U>& rhs) {
//     return {lhs.first + rhs.first, lhs.second + rhs.second};
// }
// template <typename T, typename U>
// std::pair<T, U> operator/(const std::pair<T, U>& lhs, const std::pair<T, U>& rhs) {
//     return {lhs.first/rhs.first, lhs.second/rhs.second};
// }

namespace db { // namespace
// database
class Point;
class Box;

class Chiplet;
class Pin;
class Net;
class Reticle;
class ClusterType;
class CPairType;


template<typename Derived, typename T>
class Pair2DBase {
public:
    T first() const { return static_cast<const Derived*>(this)->first(); }
    T second() const { return static_cast<const Derived*>(this)->second(); }

    T norm1() { return std::abs(first()) + std::abs(second()); }
    T norm2() { return std::sqrt(first() * first() + second() * second()); }

    T sum() { return first() + second(); }
    T mul() { return first() * second(); }
    T min() { return std::min(first(), second()); }
    T max() { return std::max(first(), second()); }
    T absmul() { return std::abs(first() * second()); }

    Derived reverse() { return Derived(second(), first()); }

    Derived operator+(T val) const { return Derived(first() + val, second() + val); }
    Derived operator-(T val) const { return Derived(first() - val, second() - val); }
    Derived operator*(T val) const { return Derived(first() * val, second() * val); }
    Derived operator/(T val) const { return Derived(first() / val, second() / val); }

    Derived operator+(const Derived& other)  const { return Derived(first() + other.first(), second() + other.second()); }
    Derived operator-(const Derived& other)  const { return Derived(first() - other.first(), second() - other.second()); }
    Derived operator*(const Derived& other)  const { return Derived(first() * other.first(), second() * other.second()); }
    Derived operator/(const Derived& other)  const { return Derived(first() / other.first(), second() / other.second()); }
    bool    operator<(const Derived& other)  const { return first() < other.first() && second() < other.second(); }
    bool    operator>(const Derived& other)  const { return first() > other.first() && second() > other.second(); }
    bool    operator<=(const Derived& other) const { return first() <= other.first() && second() <= other.second(); }
    bool    operator>=(const Derived& other) const { return first() >= other.first() && second() >= other.second(); }
    bool    operator==(const Derived& other) const { return first() == other.first() && second() == other.second(); }
    bool    operator!=(const Derived& other) const { return !(*this == other); }

    template<typename Other, typename U>
    Derived operator+(const Pair2DBase<Other, U>& other) const {
        return Derived(first() + other.first(), second() + other.second());
    }

    template<typename Other, typename U>
    Derived operator-(const Pair2DBase<Other, U>& other) const {
        return Derived(first() - other.first(), second() - other.second());
    }

    template<typename Other, typename U>
    Derived operator*(const Pair2DBase<Other, U>& other) const {
        return Derived(first() * other.first(), second() * other.second());
    }

    template<typename Other, typename U>
    Derived operator/(const Pair2DBase<Other, U>& other) const {
        return Derived(first() / other.first(), second() / other.second());
    }
};

template<typename Tr, typename T0, typename T1>
Tr overlapXY(T0* e0, T1* e1) {
    if (e0 == nullptr || e1 == nullptr) { return Tr(LOC_TYPE_MAX, LOC_TYPE_MAX); }
    LocType ox = std::max(e0->xc() + e0->w()/2, e1->xc() + e1->w()/2)
               - std::min(e0->xc() - e0->w()/2, e1->xc() - e1->w()/2) - (e0->w() + e1->w());
    LocType oy = std::max(e0->yc() + e0->h()/2, e1->yc() + e1->h()/2)
               - std::min(e0->yc() - e0->h()/2, e1->yc() - e1->h()/2) - (e0->h() + e1->h());
    return Tr(ox, oy);
}

template<typename Derived, typename T, typename Func>
Derived pairmap(const Pair2DBase<Derived, T>& p1, const Pair2DBase<Derived, T>& p2, Func f) {
    return Derived(f(p1.first(), p2.first()), f(p1.second(), p2.second()));
}
template<typename Derived1, typename Derived2, typename T1, typename T2, typename Func1, typename Func2>
Derived1 pairmap(const Pair2DBase<Derived1, T1>& p1, const Pair2DBase<Derived2, T2>& p2, Func1 f1, Func2 f2) {
    return Derived1(f1(p1.first(), p2.first()), f2(p1.second(), p2.second()));
}

bool segmentsIntersect(const Point& a, const Point& b, const Point& c, const Point& d);


class RPoint : public Pair2DBase<RPoint, RealType> {
public:
    RPoint() {}
    RPoint(RealType a_, RealType b_) : a(a_), b(b_) {}

    RealType first() const { return a; }
    RealType second() const { return b; }

    void setFirst(RealType val) { a = val; }
    void setSecond(RealType val) { b = val; }
    
    RealType a;
    RealType b;
};

class Point : public Pair2DBase<Point, LocType> {
public:
    Point() {}
    Point(LocType x_, LocType y_) : x(x_), y(y_) {}

    LocType first() const { return x; }
    LocType second() const { return y; }

    void setFirst(LocType val) { x = val; }
    void setSecond(LocType val) { y = val; }
    
    LocType x;
    LocType y;
};

class Size : public Pair2DBase<Size, LocType> {
public:
    Size() {}
    Size(LocType w_, LocType h_) : w(w_), h(h_) {}

    LocType first() const { return w; }
    LocType second() const { return h; }

    void setFirst(LocType val) { w = val; }
    void setSecond(LocType val) { h = val; }

    LocType w;
    LocType h;
};

class Num2D : public Pair2DBase<Num2D, IndexType> {
public:
    Num2D() {}
    Num2D(IndexType xPins, IndexType yPins) : nx(xPins), ny(yPins) {}

    IndexType first() const { return nx; }
    IndexType second() const { return ny; }

    void setFirst(IndexType val) { nx = val; }
    void setSecond(IndexType val) { ny = val; }

    IndexType nx;
    IndexType ny;
};


class Box {
public:
    Box() {}
    Box(IndexType id) : _id(id) {}
    Box(IndexType id, NameType name) : _id(id), _name(name) {}
    Box(LocType xc, LocType yc, LocType w, LocType h) : _loc(Point(xc, yc)), _size(Size(w, h)) {}

    void setLoc(Point p) { _loc = p; }
    void setLoc(LocType x, LocType y) { _loc.x = x; _loc.y = y; }
    void setSize(Size s) { _size = s; }
    void setSize(LocType w, LocType h) { _size.w = w; _size.h = h; }

    IndexType& id() {return _id; }
    NameType& name() {return _name; }
    LocType& xc() { return _loc.x; }
    LocType& yc() { return _loc.y; }
    LocType& w() { return _size.w; }
    LocType& h() { return _size.h; }
    Point& point() { return _loc; }
    Size& size() { return _size; }

    LocType area() { return _size.absmul(); }

    bool contains(const Point& p) const { return p.x >= _loc.x-_size.w/2 && p.x <= _loc.x+_size.w/2 && p.y >= _loc.y-_size.h/2 && p.y <= _loc.y+_size.h/2; }
    std::array<Point, 4> getCorners() const {
        return {{
            pairmap(_loc, _size/2, std::minus<LocType>{}, std::minus<LocType>{}),
            pairmap(_loc, _size/2, std::minus<LocType>{}, std::plus<LocType>{}),
            pairmap(_loc, _size/2, std::plus<LocType>{} , std::minus<LocType>{}),
            pairmap(_loc, _size/2, std::plus<LocType>{} , std::plus<LocType>{})
        }};
    }

    bool intersects(const Point& p1, const Point& p2) const {
        RealType minX = std::min(p1.x, p2.x), maxX = std::max(p1.x, p2.x);
        RealType minY = std::min(p1.y, p2.y), maxY = std::max(p1.y, p2.y);

        if (maxX < _loc.x-_size.w/2 || minX > _loc.x+_size.w/2 || maxY < _loc.y-_size.h/2 || minY > _loc.y+_size.h/2) { return false; }
            
        if (contains(p1) || contains(p2)) { return true; }
            
        auto corners = getCorners();
        for (int i = 0; i < 4; ++i) {
            Point q1 = corners[i], q2 = corners[(i + 1) % 4];
            if (segmentsIntersect(p1, p2, q1, q2)) { return true; }
        }

        return false;
    }

protected:
    IndexType _id;
    NameType _name;
    Point _loc;
    Size _size;
};

class Chiplet : public Box {
public:
    Chiplet() : Box() { chiplet_init(); }
    Chiplet(IndexType idx) : Box(idx) { chiplet_init(); }
    Chiplet(IndexType idx, NameType name) : Box(idx, name) { chiplet_init(); }

    bool& is_newgen() { return _is_newgen; }
    
    IndexType& reticle_id() {return _reticle_id; }
    RealType& theta() { return _theda; }
    RealType& theta_f() { return _theda_f; }

    IndexType argmax_z() { return std::distance(_z.begin(), std::max_element(_z.begin(), _z.end())); }
    void legalize() {
        // RealType step;
        // step = PI / 2; _theda = std::round(_theda / step) * step;
        // step = PI; _theda_f = std::round(_theda_f / step) * step;
        IndexType max_idx = argmax_z();
        _theda   = utils::LEGAL_THETA[max_idx]; _theda_f = utils::LEGAL_THETA_F[max_idx];
    }

    // IntType discrete_theta_idx() { return static_cast<IntType>(std::round(_theda * 2 / PI)) % 4; }
    // RealType discrete_theta_f_idx() { return static_cast<IntType>(std::round(_theda_f / PI)) % 2; }
    IntType discrete_theta_idx() { return argmax_z() % 4; }
    IntType discrete_theta_f_idx() { return argmax_z() / 4; }
    LocType rot_w() { return (discrete_theta_idx() % 2 == 0) ? w() : h(); }
    LocType rot_h() { return (discrete_theta_idx() % 2 == 0) ? h() : w(); }
    LocType rot_real_w() { return (discrete_theta_idx() % 2 == 0) ? w()*utils::XSCALE : h()*utils::YSCALE; }
    LocType rot_real_h() { return (discrete_theta_idx() % 2 == 0) ? h()*utils::YSCALE : w()*utils::XSCALE; }

    void setz(const std::array<RealType, utils::NUM_DISCRETE_VAR>& values) {
        std::copy(values.begin(), values.end(), _z.begin());
    }
    std::array<RealType, utils::NUM_DISCRETE_VAR>& z() { return _z; }
    RealType& z(IndexType idx) { return _z[idx]; }
    void z2theta(RealType gamma, RealType gumbel);
    
    void setNumPins(IndexType nx, IndexType ny) {_xynumPins.nx = nx; _xynumPins.ny = ny; }
    IndexType& nx_pins() {return _xynumPins.nx; }
    IndexType& ny_pins() {return _xynumPins.ny; }
    Size xy_per_pin() { return _size / _xynumPins; }

    void addPin(Pin* pin) { _vpins.emplace_back(pin); }
    std::vector<Pin*>& vpins() {return _vpins; }

    void addCoverReticle(Reticle* reticle) { _cover_reticles.emplace_back(reticle); }
    std::vector<Reticle*>& cover_reticles() { return _cover_reticles; };

    void addCluster(ClusterType* cluster) { _vclusters.emplace_back(cluster); }
    std::vector<ClusterType*>& vclusters() { return _vclusters; }

    Box& boundaryBox() { return _bdy_box; }

    std::array<LocType, 4>& space() { return _space; }

    std::array<RealType, 2+utils::NUM_DISCRETE_VAR>& grad() { return _grad; }
    void clear_grad() { _grad.fill(RealType(0)); }
    template <std::size_t N>
    void acc_grad(const std::array<RealType, N>& g) { for (IndexType idx = 0; idx < N; ++idx) { _grad[idx] += g[idx]; } }

    std::array<RealType, utils::NUM_DISCRETE_VAR>& gumbel_noise() { return _cached_gumbel_noise; }



    void print_msg(NameType path);

private:
    IndexType _reticle_id;
    bool _is_newgen;
    std::array<RealType, utils::NUM_DISCRETE_VAR> _z;
    RealType _theda;
    RealType _theda_f;
    Num2D _xynumPins;
    std::vector<Pin*> _vpins;
    std::vector<Reticle*> _cover_reticles;
    std::vector<ClusterType*> _vclusters;
    Box _bdy_box;
    std::array<LocType, 4> _space; // l, r, d, u

    std::array<RealType, 2+utils::NUM_DISCRETE_VAR> _grad;
    std::array<RealType, utils::NUM_DISCRETE_VAR> _cached_gumbel_noise;
    
    void chiplet_init();
};

class Pin : public Box {
public:
    Pin() : Box() {}
    Pin(IndexType idx) : Box(idx) {}
    Pin(IndexType idx, NameType name) : Box(idx, name) {}

    LocType& xc_offset() {return _loc.x; }
    LocType& yc_offset() {return _loc.y; }
    LocType xc() { return _chiplet->xc() + _loc.x; }
    LocType yc() { return _chiplet->yc() + _loc.y; }

    Size real_size() {
        return size() * Size(utils::XSCALE, utils::YSCALE);
    }

    Point rot_flip_real_loc() {
        Point p;
        IntType flip_idx = _chiplet->discrete_theta_f_idx();
        if      (flip_idx == 0) { p = Point( xc_offset()*utils::XSCALE, yc_offset()*utils::YSCALE); }
        else if (flip_idx == 1) { p = Point(-xc_offset()*utils::XSCALE, yc_offset()*utils::YSCALE); }

        // if (flip_idx != 0) { 
        //     std::cout << "flip_idx: " << flip_idx << std::endl; 
        //     std::cout << "(" << xc_offset() << "," << yc_offset() << ") -> " \ 
        //               << "(" << p.first()   << "," << p.second()  << ")" << std::endl;
        // }

        IntType rot_idx = _chiplet->discrete_theta_idx();
        // if (rot_idx != 0) { std::cout << "rot_idx: " << rot_idx << " (" << p.first()   << "," << p.second()  << ") -> "; }

        while (rot_idx != 0) {
            p = Point(-p.second(), p.first()); rot_idx--;
        }

        // if (_chiplet->discrete_theta_idx() != 0) { std::cout << "(" << p.first()   << "," << p.second()  << ")" << std::endl; }

        return p;
    }

    void setChiplet(Chiplet* chiplet) {_chiplet = chiplet; }
    Chiplet* owner_chiplet() { return _chiplet; }

    void setNet(Net* net) { _net = net; }
    Net* net() {return _net; }

private:
    Chiplet* _chiplet;
    Net* _net;
};

class Net {
public:
    Net() {}
    Net(IndexType idx) : _id(idx) {}
    Net(IndexType idx, NameType name) : _id(idx), _name(name) {}

    IndexType& id() {return _id; }
    NameType& name() {return _name; }
    std::vector<Pin*>& vpins() {return _vpins; }
    bool& is_power() { return _isPower; }
    void addPin(Pin* pin) { _vpins.emplace_back(pin); }

private:
    IndexType _id;
    NameType _name;
    bool _isPower = false;
    std::vector<Pin*> _vpins;
};

class Reticle : public Box {
public:
    Reticle() : Box() {}
    Reticle(IndexType idx) : Box(idx) {}
    Reticle(IndexType idx, NameType name) : Box(idx, name) {}

    void addChiplet(Chiplet* chiplet) { _vchiplets.emplace_back(chiplet); }
    std::vector<Chiplet*>& vchiplets() { return _vchiplets; }

    void print_msg(NameType path);
private:
    std::vector<Chiplet*> _vchiplets;

};

class ClusterType : public Box {
public:
    ClusterType() : Box() {}
    ClusterType(Chiplet* chiplet) : Box(), _chiplet(chiplet) { _id = _next_id++; }
    ClusterType(Chiplet* chiplet, std::vector<Pin*> vpins) : Box(), _chiplet(chiplet), _vpins(vpins) { _id = _next_id++; }

    Chiplet* chiplet() { return _chiplet; }
    std::vector<Pin*>& vpins() { return _vpins; }
    void clear_offset_center() { _loc.x = 0; _loc.y = 0; }
    Point offset_center() { return _loc; }
    Point rot_offset_center() { 
        RealType theta   = _chiplet->theta();
        RealType theta_f = _chiplet->theta_f();
        RealType cos_theta = std::cos(theta), sin_theta = std::sin(theta);
        Point loc_f = _loc * RPoint(std::cos(theta_f), 1);
        return loc_f * RPoint(1, 1) * cos_theta + loc_f.reverse() * RPoint(-1, 1) * sin_theta;
    }
    Size bdy_wh() { return _size; }

    void calculate_center() { 
        clear_offset_center();

        IndexType num_pins = _vpins.size();
        if (num_pins == 0) { return; } 

        for (Pin* pin : _vpins) { _loc = _loc + pin->point(); } 
        _loc = _loc / _vpins.size();
    }

    void calculate_wh() { 
        Point _loc_ll = Point(LOC_TYPE_MAX, LOC_TYPE_MAX);
        Point _loc_hh = Point(LOC_TYPE_MIN, LOC_TYPE_MIN);
        for (Pin* pin : _vpins) {
            _loc_ll = pairmap(_loc_ll, pin->point(),  [](LocType a, LocType b) { return std::min(a, b); });
            _loc_hh = pairmap(_loc_hh, pin->point(),  [](LocType a, LocType b) { return std::max(a, b); });
        }
        _size = Size(0, 0) + _loc_hh - _loc_ll;
    }

    void setConnCluster(ClusterType* conn_cluster) { _conn_cluster = conn_cluster; }
    ClusterType* connCluster() { return _conn_cluster; }

    LocType pin_area() { 
        LocType num_pin = _chiplet->is_newgen() ? _virtual_num_pins : _vpins.size(); 
        return num_pin * _chiplet->xy_per_pin().absmul();
    }

    IndexType& virtual_num_pins() { return _virtual_num_pins; }

private:
    static IndexType _next_id;
    Chiplet* _chiplet;
    std::vector<Pin*> _vpins;
    ClusterType* _conn_cluster;

    IndexType _virtual_num_pins = 0;
};


class CPairType {
public:
    CPairType(Chiplet* chiplet0, Chiplet* chiplet1) : _id(_next_id++) {
        if (chiplet0->id() == chiplet1->id()) {
            throw std::invalid_argument("there are some bug to be fixed in clusterpairtype");
        }
        _chiplet0 = chiplet0->id() < chiplet1->id() ? chiplet0 : chiplet1;
        _chiplet1 = chiplet0->id() > chiplet1->id() ? chiplet0 : chiplet1;
    }
    IndexType& id() { return _id; }
    bool& is_critical() { return _is_critical; }

    bool addPinPair(Pin* pin0, Pin* pin1);

    Chiplet* chiplet0() { return _chiplet0; }
    Chiplet* chiplet1() { return _chiplet1; }
    std::vector<Pin*> pins0() { return _vpins0; }
    std::vector<Pin*> pins1() { return _vpins1; }

    void setChipletPair(Chiplet* c0, Chiplet*c1) { _chiplet0 = c0; _chiplet1 = c1; }
    void setClusterPair(ClusterType* cl0, ClusterType*cl1) { _cluster0 = cl0; _cluster1 = cl1; }
    ClusterType* cluster0() { return _cluster0; }
    ClusterType* cluster1() { return _cluster1; }

    std::vector<Path>& v_paths() { return _v_paths; }

    void print_msg(NameType path);

private:
    IndexType _id;
    static IndexType _next_id;
    bool _is_critical;

    Chiplet* _chiplet0;
    Chiplet* _chiplet1;
    std::vector<Pin*> _vpins0;
    std::vector<Pin*> _vpins1;

    ClusterType* _cluster0; 
    ClusterType* _cluster1; 

    std::vector<Path> _v_paths;
};

class Database {
public:
    Chiplet*     allocateChiplet()                                          { return allocateToVector(_vchiplets); }
    Chiplet*     allocateChiplet(IndexType idx)                             { return allocateToVector(_vchiplets, idx); }
    Chiplet*     allocateChiplet(IndexType idx, NameType name)              { return allocateToVector(_vchiplets, idx, name); }
    Pin*         allocatePin()                                              { return allocateToVector(_vpins); }
    Pin*         allocatePin(IndexType idx)                                 { return allocateToVector(_vpins, idx); }
    Pin*         allocatePin(IndexType idx, NameType name)                  { return allocateToVector(_vpins, idx, name); }
    Net*         allocateNet()                                              { return allocateToVector(_vnets); }
    Net*         allocateNet(IndexType idx)                                 { return allocateToVector(_vnets, idx); }
    Net*         allocateNet(IndexType idx, NameType name)                  { return allocateToVector(_vnets, idx, name); }
    Reticle*     allocateReticle()                                          { return allocateToVector(_vreticles); }
    Reticle*     allocateReticle(IndexType idx)                             { return allocateToVector(_vreticles, idx); }
    Reticle*     allocateReticle(IndexType idx, NameType name)              { return allocateToVector(_vreticles, idx, name); }
    CPairType*   allocateCPair(Chiplet* chiplet0, Chiplet* chiplet1)        { return allocateToVector(_vcpairs, chiplet0, chiplet1); }
    ClusterType* allocateCluster(Chiplet* chiplet)                          { return allocateToVector(_vclusters, chiplet); }
    ClusterType* allocateCluster(Chiplet* chiplet, std::vector<Pin*> vpins) { return allocateToVector(_vclusters, chiplet, vpins); }

    std::vector<std::unique_ptr<Chiplet>>&     chiplets()  { return _vchiplets; }
    std::vector<std::unique_ptr<Pin>>&         pins()      { return _vpins; }
    std::vector<std::unique_ptr<Net>>&         nets()      { return _vnets; }
    std::vector<std::unique_ptr<Reticle>>&     reticles()  { return _vreticles; }
    std::vector<std::unique_ptr<ClusterType>>& clusters()  { return _vclusters; }
    std::vector<std::unique_ptr<CPairType>>&   cpairs()    { return _vcpairs; }

    LocMatrix& conn_mat() { return _connectivity_matrix; }
    LocMatrix& weight_mat() { return _weight_matrix; }

    template<typename T>
    T* getElementWithId(std::vector<std::unique_ptr<T>>& vec, IndexType idx) {
        if (idx < vec.size() && vec[idx]->id() == idx) { return vec[idx].get(); }
        for (auto& item : vec) { if (item->id() == idx) { return item.get(); } }
        return nullptr;
    }

    Chiplet*  getChipletwithId(IndexType idx) { return getElementWithId(_vchiplets, idx); }
    Pin*      getPinwithId(IndexType idx)     { return getElementWithId(_vpins, idx); }
    Net*      getNetwithId(IndexType idx)     { return getElementWithId(_vnets, idx); }
    Reticle*  getReticlewithId(IndexType idx) { return getElementWithId(_vreticles, idx); }

    void findallChipletCoverReticles(IndexType id);

private:
    template <typename Vec, typename... Args>
    typename Vec::value_type::element_type* allocateToVector(Vec& vec, Args&&... args) {
        auto ptr = std::make_unique<typename Vec::value_type::element_type>(std::forward<Args>(args)...);
        auto raw = ptr.get();
        vec.emplace_back(std::move(ptr));
        return raw;
    }

    std::vector<std::unique_ptr<Chiplet>>     _vchiplets;
    std::vector<std::unique_ptr<Pin>>         _vpins;
    std::vector<std::unique_ptr<Net>>         _vnets;
    std::vector<std::unique_ptr<Reticle>>     _vreticles;
    std::vector<std::unique_ptr<CPairType>>   _vcpairs;
    std::vector<std::unique_ptr<ClusterType>> _vclusters;

    LocMatrix _connectivity_matrix;
    LocMatrix _weight_matrix;
};

// class Database {
//     public:
//         Database() {}
//         Chiplet*     allocateChiplet()                                          { return &_vchiplets.emplace_back(Chiplet()); }
//         Chiplet*     allocateChiplet(IndexType idx)                             { return &_vchiplets.emplace_back(Chiplet(idx)); }
//         Chiplet*     allocateChiplet(IndexType idx, NameType name)              { return &_vchiplets.emplace_back(Chiplet(idx, name)); }
//         Pin*         allocatePin()                                              { return &_vpins.emplace_back(Pin()); }
//         Pin*         allocatePin(IndexType idx)                                 { return &_vpins.emplace_back(Pin(idx)); }
//         Pin*         allocatePin(IndexType idx, NameType name)                  { return &_vpins.emplace_back(Pin(idx, name)); }
//         Net*         allocateNet()                                              { return &_vnets.emplace_back(Net()); }
//         Net*         allocateNet(IndexType idx)                                 { return &_vnets.emplace_back(Net(idx)); }
//         Net*         allocateNet(IndexType idx, NameType name)                  { return &_vnets.emplace_back(Net(idx, name)); }
//         Reticle*     allocateReticle()                                          { return &_vreticles.emplace_back(Reticle()); }
//         Reticle*     allocateReticle(IndexType idx)                             { return &_vreticles.emplace_back(Reticle(idx)); }
//         Reticle*     allocateReticle(IndexType idx, NameType name)              { return &_vreticles.emplace_back(Reticle(idx, name)); }
//         CPairType*   allocateCPair(Chiplet* chiplet0, Chiplet* chiplet1)        { return &_vcpairs.emplace_back(CPairType(chiplet0, chiplet1)); }
//         ClusterType* allocateCluster(Chiplet* chiplet, std::vector<Pin*> vpins) { return &_vclusters.emplace_back(ClusterType(chiplet, vpins)); }

//         std::vector<Chiplet>&   chiplets() { return _vchiplets; }
//         std::vector<Pin>&       pins()     { return _vpins; }
//         std::vector<Net>&       nets()     { return _vnets; }
//         std::vector<Reticle>&  reticles() { return _vreticles; }
//         std::vector<ClusterType>& clusters()   { return _vclusters; }
//         std::vector<CPairType>& cpairs()   { return _vcpairs; }
        

//         template<typename T>
//         T* getElementWithId(std::vector<T>& vec, IndexType idx) {
//             if (idx < vec.size() && vec[idx].id() == idx) { return &vec[idx]; }
//             for (auto& item : vec) { if (item.id() == idx) { return &item; } }
//             return nullptr;
//         }
//         Chiplet*  getChipletwithId(IndexType idx) { return getElementWithId(_vchiplets, idx); }
//         Pin*      getPinwithId(IndexType idx)     { return getElementWithId(_vpins, idx); }
//         Net*      getNetwithId(IndexType idx)     { return getElementWithId(_vnets, idx); }
//         Reticle* getReticlewithId(IndexType idx) { return getElementWithId(_vreticles, idx); }

//         void findallChipletCoverReticles();

//     private:
//         std::vector<Chiplet> _vchiplets;
//         std::vector<Pin> _vpins;
//         std::vector<Net> _vnets;
//         std::vector<Reticle> _vreticles;
//         std::vector<CPairType> _vcpairs;
//         std::vector<ClusterType> _vclusters;
        
// };




}; // end namespace

#endif



