//
// Created by salmon on 17-11-1.
//
#include "gBoundedCurve.h"
#include <simpla/data/Serializable.h>

namespace simpla {
namespace geometry {

void gBoundedCurve2D::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
}
std::shared_ptr<simpla::data::DataEntry> gBoundedCurve2D::Serialize() const {
    auto res = base_type::Serialize();
    //    res->SetValue("Data", m_data_);
    return res;
}

void gBoundedCurve2D::Open() { UNIMPLEMENTED; };
void gBoundedCurve2D::Close() { UNIMPLEMENTED; };
bool gBoundedCurve2D::IsClosed() const { return false; }
size_type gBoundedCurve2D::size() const { return m_data_.size(); };
point2d_type gBoundedCurve2D::GetPoint2D(index_type s) const { return m_data_[(s + m_data_.size()) % m_data_.size()]; }
void gBoundedCurve2D::AddPoint2D(Real x, Real y) { m_data_.emplace_back(point2d_type{x, y}); }
point2d_type gBoundedCurve2D::xy(Real u) const {
    point2d_type res{SP_SNaN, SP_SNaN};
    index_type s = static_cast<index_type>(u);
    Real r = u - s;
    auto n = m_data_.size();
    index_type s0 = s, s1 = s + 1;
    if (s < n || IsClosed()) {
        s0 = (s0 + n) % n;
        s1 = (s1 + n) % n;
    } else if (s < 0) {
        s1 = 0;
        s0 = 1;
        r = (-u + 1);
    } else if (s >= n) {
        s0 = n - 2;
        s1 = n - 1;
        r = u - (n - 1);
    }
    res = (1 - r) * m_data_[s0] + r * m_data_[s1];
    return res;
};
/**************************************************************/

// gBoundedCurve::gBoundedCurve(gBoundedCurve const &other) : ggBoundedCurve(other) {
//    std::vector<point_type>(other.m_data_).swap(m_data_);
//};
void gBoundedCurve::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
}
std::shared_ptr<simpla::data::DataEntry> gBoundedCurve::Serialize() const { return base_type::Serialize(); }
void gBoundedCurve::Open() { UNIMPLEMENTED; };
void gBoundedCurve::Close() { UNIMPLEMENTED; };
bool gBoundedCurve::IsClosed() const { return false; }
void gBoundedCurve::AddPoint(point_type const &p) { m_data_.push_back(p); };
point_type gBoundedCurve::GetPoint(index_type s) const { return m_data_[(s + m_data_.size()) % m_data_.size()]; };
point_type gBoundedCurve::xyz(Real u) const {
    point_type res{SP_SNaN, SP_SNaN, SP_SNaN};
    index_type s = static_cast<index_type>(u);
    Real r = u - s;
    auto n = m_data_.size();
    index_type s0 = s, s1 = s + 1;
    if (s < n || IsClosed()) {
        s0 = (s0 + n) % n;
        s1 = (s1 + n) % n;
    } else if (s < 0) {
        s1 = 0;
        s0 = 1;
        r = (-u + 1);
    } else if (s >= n) {
        s0 = n - 2;
        s1 = n - 1;
        r = u - (n - 1);
    }
    res = (1 - r) * m_data_[s0] + r * m_data_[s1];
    return res;
};

}  // namespace geometry{
}  // namespace simpla{