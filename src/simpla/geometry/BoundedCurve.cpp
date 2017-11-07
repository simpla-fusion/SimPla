//
// Created by salmon on 17-11-1.
//

#include "BoundedCurve.h"

namespace simpla {
namespace geometry {

BoundedCurve::BoundedCurve() = default;
BoundedCurve::BoundedCurve(BoundedCurve const &other) = default;
BoundedCurve::~BoundedCurve() = default;
BoundedCurve::BoundedCurve(Axis const &axis) : Curve(axis){};
void BoundedCurve::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> BoundedCurve::Serialize() const { return base_type::Serialize(); }

struct BoundedCurve2D::pimpl_s {
    std::vector<point2d_type> m_data_;
};
BoundedCurve2D::BoundedCurve2D() : m_pimpl_(new pimpl_s){};
BoundedCurve2D::BoundedCurve2D(BoundedCurve2D const &other) : base_type(other), m_pimpl_(new pimpl_s) {
    std::vector<point2d_type>(other.m_pimpl_->m_data_).swap(m_pimpl_->m_data_);
};
BoundedCurve2D::~BoundedCurve2D() { delete m_pimpl_; }
BoundedCurve2D::BoundedCurve2D(Axis const &axis) : BoundedCurve(axis), m_pimpl_(new pimpl_s){};
void BoundedCurve2D::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> BoundedCurve2D::Serialize() const { return base_type::Serialize(); }
point_type BoundedCurve2D::xyz(Real u) const {
    UNIMPLEMENTED;
    return point_type{};
};
void BoundedCurve2D::Open() { UNIMPLEMENTED; };
void BoundedCurve2D::Close() { UNIMPLEMENTED; };
bool BoundedCurve2D::IsClosed() const { return false; }
size_type BoundedCurve2D::size() const { return m_pimpl_->m_data_.size(); };
void BoundedCurve2D::AddPoint(point_type const &p) { AddPoint(p[0], p[1]); };
point_type BoundedCurve2D::GetPoint(index_type s) const {
    auto p = GetPoint2d(s);
    return m_axis_.xyz(p[0], p[1], 0);
};
point2d_type BoundedCurve2D::GetPoint2d(index_type s) const {
    return m_pimpl_->m_data_[(s + m_pimpl_->m_data_.size()) % m_pimpl_->m_data_.size()];
}
void BoundedCurve2D::AddPoint(Real x, Real y) { m_pimpl_->m_data_.emplace_back(point2d_type{x, y}); }
std::vector<point2d_type> &BoundedCurve2D::data() { return m_pimpl_->m_data_; }
std::vector<point2d_type> const &BoundedCurve2D::data() const { return m_pimpl_->m_data_; }
/**************************************************************/

struct BoundedCurve3D::pimpl_s {
    std::vector<point_type> m_data_;
};
BoundedCurve3D::BoundedCurve3D() : BoundedCurve(), m_pimpl_(new pimpl_s){};
BoundedCurve3D::BoundedCurve3D(BoundedCurve3D const &other) : BoundedCurve(other), m_pimpl_(new pimpl_s) {
    std::vector<point_type>(other.m_pimpl_->m_data_).swap(m_pimpl_->m_data_);
};
BoundedCurve3D::~BoundedCurve3D() { delete m_pimpl_; }
BoundedCurve3D::BoundedCurve3D(Axis const &axis) : BoundedCurve(axis), m_pimpl_(new pimpl_s){};
void BoundedCurve3D::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> BoundedCurve3D::Serialize() const { return base_type::Serialize(); }
point_type BoundedCurve3D::xyz(Real u) const {
    UNIMPLEMENTED;
    return point_type{};
};
void BoundedCurve3D::Open() { UNIMPLEMENTED; };
void BoundedCurve3D::Close() { UNIMPLEMENTED; };
bool BoundedCurve3D::IsClosed() const { return false; }
size_type BoundedCurve3D::size() const { return m_pimpl_->m_data_.size(); };
void BoundedCurve3D::AddPoint(Real x, Real y, Real z) { m_pimpl_->m_data_.emplace_back(point_type{x, y, z}); }
void BoundedCurve3D::AddPoint(point_type const &p) { AddPoint(p[0], p[1], p[2]); };
point_type BoundedCurve3D::GetPoint(index_type s) const {
    return m_axis_.xyz(m_pimpl_->m_data_[(s + m_pimpl_->m_data_.size()) % m_pimpl_->m_data_.size()]);
};

std::vector<point_type> &BoundedCurve3D::data() { return m_pimpl_->m_data_; }
std::vector<point_type> const &BoundedCurve3D::data() const { return m_pimpl_->m_data_; }
}  // namespace geometry{
}  // namespace simpla{