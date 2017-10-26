/**
 * @file polygon.cpp
 * @author salmon
 * @date 2015-11-18.
 */

#include "simpla/SIMPLA_config.h"

#include "simpla/algebra/nTuple.h"

#include "GeoAlgorithm.h"
#include "Polygon.h"

namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(Polygon)

struct Polygon::pimpl_s {
    typedef nTuple<Real, 2> point2d_type;

    std::vector<point2d_type> m_polygon_;
    std::vector<Real> constant_;
    std::vector<Real> multiple_;

    std::vector<point2d_type> &data() { return m_polygon_; };
    std::vector<point2d_type> const &data() const { return m_polygon_; };
    void push_back(point2d_type const &p);
    void deploy();
    Real nearest_point(Real *x, Real *y) const;
    bool check_inside(Real x, Real y) const;

    point2d_type m_min_{0, 0};
    point2d_type m_max_{0, 0};
};
Polygon::Polygon() : m_pimpl_(new pimpl_s) {}
Polygon::~Polygon() { delete m_pimpl_; }
Polygon::Polygon(Polygon const &) : m_pimpl_(new pimpl_s) {}
Polygon::Polygon(Axis const &axis) : Surface(axis), m_pimpl_(new pimpl_s) {}
void Polygon::Open() {}
void Polygon::Close() {}
void Polygon::push_back(Real u, Real v) {}
void Polygon::push_back(size_type num, Real const *u, Real const *v) {}
std::shared_ptr<data::DataNode> Polygon::Serialize() const {
    auto res = base_type::Serialize();
    //        auto v_array = data::DataArrayWrapper<point2d_type>::New();
    //        for (size_type s = 0, se = m_polygon_.size(); s < se; ++s) { v_array->Add(m_polygon_[s]); }
    //        tdb->Set("data", std::dynamic_pointer_cast<data::DataNode>(v_array));
    return res;
};
void Polygon::Deserialize(std::shared_ptr<data::DataNode> const &tdb) { base_type::Deserialize(tdb); }

std::tuple<bool, bool> Polygon::IsClosed() const { return std::make_tuple(false, false); }
std::tuple<bool, bool> Polygon::IsPeriodic() const { return std::make_tuple(false, false); }
nTuple<Real, 2> Polygon::GetPeriod() const { return nTuple<Real, 2>{SP_INFINITY, SP_INFINITY}; }
nTuple<Real, 2> Polygon::GetMinParameter() const { return nTuple<Real, 2>{-SP_INFINITY, -SP_INFINITY}; }
nTuple<Real, 2> Polygon::GetMaxParameter() const { return nTuple<Real, 2>{SP_INFINITY, SP_INFINITY}; }

int Polygon::CheckOverlap(box_type const &) const { return 0; }
int Polygon::FindIntersection(std::shared_ptr<const GeoObject> const &, std::vector<Real> &, Real tolerance) const {
    return 0;
}
point_type Polygon::Value(Real u, Real v) const { return point_type{0, 0, 0}; }
Real Polygon::pimpl_s::nearest_point(Real *x, Real *y) const {
    typedef nTuple<Real, 2> Vec2;

    point2d_type x0;

    x0[0] = *x;
    x0[1] = *y;

    point2d_type p0;  // = m_polygon_.back();

    point2d_type p1;

    auto it = m_polygon_.begin();

    Real d2 = std::numeric_limits<Real>::max();

    while (it != m_polygon_.end()) {
        p1 = *it;
        ++it;

        Vec2 u, v;

        u = x0 - p0;
        v = p1 - p0;

        Real v2 = dot(v, v);

        Real s = dot(u, v) / v2;

        point2d_type p;

        if (s < 0) {
            s = 0;
        } else if (s > 1) {
            s = 1;
        }

        p = ((1 - s) * p0 + s * p1);
        /**
         * if \f$ v \times u \cdot e_z >0 \f$ then `in` else `out`
         */
        UNIMPLEMENTED;

        Real dd = 0;  // dot(x0 - p, x0 - p);

        if (std::abs(dd) < std::abs(d2)) {
            d2 = dd;
            (*x) = x0[0] - u[0];
            (*y) = x0[1] - u[1];
        }
        p0 = p1;
    }

    d2 = std::sqrt(d2);

    return check_inside(*x, *y) > 0 ? d2 : -d2;
}

void Polygon::pimpl_s::push_back(point2d_type const &p) { m_polygon_.push_back(p); }

void Polygon::pimpl_s::deploy() {
    size_t num_of_vertex_ = m_polygon_.size();
    constant_.resize(num_of_vertex_);
    multiple_.resize(num_of_vertex_);

    for (size_t i = 0, j = num_of_vertex_ - 1; i < num_of_vertex_; i++) {
        if (m_polygon_[j][1] == m_polygon_[i][1]) {
            constant_[i] = m_polygon_[i][0];
            multiple_[i] = 0;
        } else {
            constant_[i] = m_polygon_[i][0] -
                           (m_polygon_[i][1] * m_polygon_[j][0]) / (m_polygon_[j][1] - m_polygon_[i][1]) +
                           (m_polygon_[i][1] * m_polygon_[i][0]) / (m_polygon_[j][1] - m_polygon_[i][1]);
            multiple_[i] = (m_polygon_[j][0] - m_polygon_[i][0]) / (m_polygon_[j][1] - m_polygon_[i][1]);
        }
        j = i;
    }

    m_min_ = m_polygon_.front();
    m_max_ = m_polygon_.front();

    for (auto const &p : m_polygon_) { geometry::extent_box(&m_min_, &m_max_, &p[0]); }
}

bool Polygon::pimpl_s::check_inside(Real x, Real y) const {
    if ((x >= m_min_[0]) && (y >= m_min_[1]) && (x < m_max_[0]) && (y < m_max_[1])) {
        size_t num_of_vertex_ = m_polygon_.size();

        bool oddNodes = false;

        for (size_t i = 0, j = num_of_vertex_ - 1; i < num_of_vertex_; i++) {
            if (((m_polygon_[i][1] < y) && (m_polygon_[j][1] >= y)) ||
                ((m_polygon_[j][1] < y) && (m_polygon_[i][1] >= y))) {
                oddNodes ^= (y * multiple_[i] + constant_[i] < x);
            }
            j = i;
        }

        return oddNodes;
    } else {
        return false;
    }
}
}
}  // namespace simpla//namespace  geometry