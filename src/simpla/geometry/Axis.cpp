//
// Created by salmon on 17-10-23.
//
#include "Axis.h"
#include "spLine.h"
#include "spPlane.h"
namespace simpla {
namespace geometry {

void Axis::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    m_origin_ = cfg->GetValue("Origin", m_origin_);
    m_axis_ = cfg->GetValue("Axis", m_axis_);
}
std::shared_ptr<simpla::data::DataEntry> Axis::Serialize() const {
    auto res = simpla::data::DataEntry::New(simpla::data::DataEntry::DN_TABLE);
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
Axis Axis::Moved(const point_type &p) const {
    Axis res(*this);
    res.Move(p);
    return std::move(res);
}

std::shared_ptr<spPlane> Axis::GetPlane(int n) const {
    return spPlane::New(Axis{o, m_axis_[(n + 1) % 3], m_axis_[(n + 2) % 3]});
}
std::shared_ptr<spPlane> Axis::GetPlaneXY() const { return GetPlane(2); }
std::shared_ptr<spPlane> Axis::GetPlaneYZ() const { return GetPlane(1); }
std::shared_ptr<spPlane> Axis::GetPlaneZX() const { return GetPlane(0); }
std::shared_ptr<spLine> Axis::GetAxe(int n) const { return spLine::New(Axis{o, m_axis_[n]}); }
std::shared_ptr<spLine> Axis::GetPlaneX() const { return GetAxe(0); }
std::shared_ptr<spLine> Axis::GetPlaneY() const { return GetAxe(0); }
std::shared_ptr<spLine> Axis::GetPlaneZ() const { return GetAxe(0); }

}  // namespace geometry{
}  // namespace simpla{