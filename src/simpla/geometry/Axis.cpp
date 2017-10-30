//
// Created by salmon on 17-10-23.
//
#include "Axis.h"
#include "Line.h"
#include "Plane.h"
namespace simpla {
namespace geometry {

void Axis::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {
    m_origin_ = cfg->GetValue("Origin", m_origin_);
    m_axis_ = cfg->GetValue("Axis", m_axis_);
}
std::shared_ptr<simpla::data::DataNode> Axis::Serialize() const {
    auto res = simpla::data::DataNode::New(simpla::data::DataNode::DN_TABLE);
    res->SetValue("Origin", m_origin_);
    res->SetValue("Axis", m_axis_);
    return res;
}
void Axis::Mirror(const point_type &p) { UNIMPLEMENTED; }
void Axis::Mirror(const Axis &a1) { UNIMPLEMENTED; }
void Axis::Rotate(const Axis &a1, Real angle) { UNIMPLEMENTED; }
void Axis::Scale(Real s, int dir) {
    if (dir < 0) {
        m_axis_ *= s;
    } else {
        m_axis_[dir % 3] *= s;
    }
}
void Axis::Translate(const vector_type &v) { m_origin_ += v; }
void Axis::Move(const point_type &p) { m_origin_ = p; }

std::shared_ptr<Plane> Axis::GetPlane(int n) const {
    return Plane::New(Axis{o, m_axis_[(n + 1) % 3], m_axis_[(n + 2) % 3]});
}
std::shared_ptr<Plane> Axis::GetPlaneXY() const { return GetPlane(2); }
std::shared_ptr<Plane> Axis::GetPlaneYZ() const { return GetPlane(1); }
std::shared_ptr<Plane> Axis::GetPlaneZX() const { return GetPlane(0); }
std::shared_ptr<Line> Axis::GetAxe(int n) const { return Line::New(Axis{o, m_axis_[n]}); }
std::shared_ptr<Line> Axis::GetPlaneX() const { return GetAxe(0); }
std::shared_ptr<Line> Axis::GetPlaneY() const { return GetAxe(0); }
std::shared_ptr<Line> Axis::GetPlaneZ() const { return GetAxe(0); }

}  // namespace geometry{
}  // namespace simpla{