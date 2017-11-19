//
// Created by salmon on 17-11-1.
//

#include "BoundedCurve.h"

namespace simpla {
namespace geometry {

void BoundedCurve2D::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
}
std::shared_ptr<simpla::data::DataEntry> BoundedCurve2D::Serialize() const {
    auto res = base_type::Serialize();
    UNIMPLEMENTED;
    //    res->SetValue("Data", m_data_);
    return res;
}

void BoundedCurve2D::Open() { UNIMPLEMENTED; };
void BoundedCurve2D::Close() { UNIMPLEMENTED; };
bool BoundedCurve2D::IsClosed() const { return false; }
size_type BoundedCurve2D::size() const { return m_data_.size(); };
point2d_type BoundedCurve2D::GetPoint2D(index_type s) const { return m_data_[(s + m_data_.size()) % m_data_.size()]; }
void BoundedCurve2D::AddPoint2D(Real x, Real y) { m_data_.emplace_back(point2d_type{x, y}); }
/**************************************************************/

// BoundedCurve3D::BoundedCurve3D(BoundedCurve3D const &other) : BoundedCurve(other) {
//    std::vector<point_type>(other.m_data_).swap(m_data_);
//};
void BoundedCurve3D::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
}
std::shared_ptr<simpla::data::DataEntry> BoundedCurve3D::Serialize() const { return base_type::Serialize(); }

void BoundedCurve3D::Open() { UNIMPLEMENTED; };
void BoundedCurve3D::Close() { UNIMPLEMENTED; };
bool BoundedCurve3D::IsClosed() const { return false; }
void BoundedCurve3D::AddPoint(point_type const &p) { m_data_.push_back(p); };
point_type BoundedCurve3D::GetPoint(index_type s) const { return m_data_[(s + m_data_.size()) % m_data_.size()]; };

}  // namespace geometry{
}  // namespace simpla{