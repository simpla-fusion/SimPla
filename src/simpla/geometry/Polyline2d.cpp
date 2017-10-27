//
// Created by salmon on 17-10-27.
//
#include "Polyline2d.h"
#include <simpla/SIMPLA_config.h>
#include <simpla/algebra/nTuple.h>
#include "GeoAlgorithm.h"

namespace simpla {
template <typename, int...>
struct nTuple;
namespace geometry {
struct Polyline2d::pimpl_s {
    typedef nTuple<Real, 2> point2d_type;
    std::vector<point2d_type> m_uv_;
    std::vector<Real> constant_;
    std::vector<Real> multiple_;

    void SetUp();
    Real nearest_point(Real *x, Real *y) const;
    bool check_inside(Real x, Real y) const;

    point2d_type m_min_{0, 0};
    point2d_type m_max_{0, 0};
};
Polyline2d::Polyline2d() : m_pimpl_(new pimpl_s) {}
Polyline2d::~Polyline2d() { delete m_pimpl_; }
Polyline2d::Polyline2d(Polyline2d const &) : m_pimpl_(new pimpl_s) {}
Polyline2d::Polyline2d(Axis const &axis) : Curve(axis), m_pimpl_(new pimpl_s) {}
 std::shared_ptr<data::DataNode> Polyline2d::Serialize() const {
    auto res = base_type::Serialize();
    //        auto v_array = data::DataArrayWrapper<point2d_type>::New();
    //        for (size_type s = 0, se = m_uv_.size(); s < se; ++s) { v_array->Add(m_uv_[s]); }
    //        tdb->Set("data", std::dynamic_pointer_cast<data::DataNode>(v_array));
    return res;
};
void Polyline2d::Deserialize(std::shared_ptr<data::DataNode> const &tdb) { base_type::Deserialize(tdb); }

bool Polyline2d::IsClosed() const { return false; }
bool Polyline2d::IsPeriodic() const { return false; }
Real Polyline2d::GetPeriod() const { return SP_INFINITY; }
Real Polyline2d::GetMinParameter() const { return -SP_INFINITY; }
Real Polyline2d::GetMaxParameter() const { return -SP_INFINITY; }

void Polyline2d::AddUV(Real u, Real v) { m_pimpl_->m_uv_.emplace_back(nTuple<Real,2>{u, v}); }
void Polyline2d::AddPoint(point_type const &p, Real tolerance) {
    auto uvw = m_axis_.uvw(p);
    ASSERT(std::abs(uvw[2]) < tolerance);
    AddUV(uvw[0], uvw[1]);
}
point_type Polyline2d::StartPoint(int n) const {
    auto uv = m_pimpl_->m_uv_[n % (m_pimpl_->m_uv_.size())];
    return m_axis_.xyz(uv[0], uv[1], 0);
}
point_type Polyline2d::EndPoint(int n) const {
    auto uv = m_pimpl_->m_uv_[(n + 1) % (m_pimpl_->m_uv_.size())];
    return m_axis_.xyz(uv[0], uv[1], 0);
}
size_type Polyline2d::size() const { return m_pimpl_->m_uv_.size(); }
std::vector<nTuple<Real, 2>> &Polyline2d::data() { return m_pimpl_->m_uv_; }
std::vector<nTuple<Real, 2>> const &Polyline2d::data() const { return m_pimpl_->m_uv_; }
point_type Polyline2d::Value(Real u) const {
    point_type res{SP_SNaN, SP_SNaN, SP_SNaN};
    auto num = static_cast<int>(size());
    if ((u >= 0 && u < num) || IsClosed()) {
        auto n = static_cast<int>(u);
        auto r = u - n;
        res = (1.0 - r) * StartPoint(n) + r * EndPoint(n);
    } else if (u < 0 && num > 1) {
        res = StartPoint(0) + u * (StartPoint(0) - StartPoint(1));
    } else if (u > num && u >= 2) {
        res = StartPoint(num - 1) + (u - num) * (StartPoint(num - 1) - StartPoint(num - 2));
    } else {
        UNIMPLEMENTED;
    }
    return res;
}
Real Polyline2d::pimpl_s::nearest_point(Real *x, Real *y) const {
    typedef nTuple<Real, 2> Vec2;
    point2d_type x0{SP_SNaN, SP_SNaN};
    x0[0] = *x;
    x0[1] = *y;
    point2d_type p0{SP_SNaN, SP_SNaN};  // = m_uv_.back();
    point2d_type p1{SP_SNaN, SP_SNaN};
    auto it = m_uv_.begin();
    Real d2 = std::numeric_limits<Real>::max();
    while (it != m_uv_.end()) {
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
void Polyline2d::Close() { m_pimpl_->SetUp(); }

void Polyline2d::pimpl_s::SetUp() {
    size_t num_of_vertex_ = m_uv_.size();
    constant_.resize(num_of_vertex_);
    multiple_.resize(num_of_vertex_);

    for (size_t i = 0, j = num_of_vertex_ - 1; i < num_of_vertex_; i++) {
        if (m_uv_[j][1] == m_uv_[i][1]) {
            constant_[i] = m_uv_[i][0];
            multiple_[i] = 0;
        } else {
            constant_[i] = m_uv_[i][0] - (m_uv_[i][1] * m_uv_[j][0]) / (m_uv_[j][1] - m_uv_[i][1]) +
                           (m_uv_[i][1] * m_uv_[i][0]) / (m_uv_[j][1] - m_uv_[i][1]);
            multiple_[i] = (m_uv_[j][0] - m_uv_[i][0]) / (m_uv_[j][1] - m_uv_[i][1]);
        }
        j = i;
    }

    m_min_ = m_uv_.front();
    m_max_ = m_uv_.front();

    for (auto const &p : m_uv_) { geometry::extent_box(&m_min_, &m_max_, &p[0]); }
}

bool Polyline2d::pimpl_s::check_inside(Real x, Real y) const {
    if ((x >= m_min_[0]) && (y >= m_min_[1]) && (x < m_max_[0]) && (y < m_max_[1])) {
        size_t num_of_vertex_ = m_uv_.size();
        bool oddNodes = false;
        for (size_t i = 0, j = num_of_vertex_ - 1; i < num_of_vertex_; i++) {
            if (((m_uv_[i][1] < y) && (m_uv_[j][1] >= y)) || ((m_uv_[j][1] < y) && (m_uv_[i][1] >= y))) {
                oddNodes ^= (y * multiple_[i] + constant_[i] < x);
            }
            j = i;
        }
        return oddNodes;
    } else {
        return false;
    }
}

}  // namespace geometry{
}  // namespace simpla{