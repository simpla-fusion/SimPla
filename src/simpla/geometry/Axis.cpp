//
// Created by salmon on 17-10-23.
//
#include "Axis.h"
#include <simpla/data/Serializable.h>
namespace simpla {
namespace geometry {
Axis::Axis(Axis const &other) : data::Configurable(other) {}
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
void Axis::Translate(const vector_type &v) { m_Origin_ += v; }
void Axis::Move(const point_type &p) { m_Origin_ = p; }
Axis Axis::Moved(const point_type &p) const {
    Axis res(*this);
    res.Move(p);
    return std::move(res);
}
//
// std::shared_ptr<spPlane> Axis::GetPlane(int n) const {
//    return spPlane::New(Axis{o, m_axis_[(n + 1) % 3], m_axis_[(n + 2) % 3]});
//}
// std::shared_ptr<spPlane> Axis::GetPlaneXY() const { return GetPlane(2); }
// std::shared_ptr<spPlane> Axis::GetPlaneYZ() const { return GetPlane(1); }
// std::shared_ptr<spPlane> Axis::GetPlaneZX() const { return GetPlane(0); }
// std::shared_ptr<spLine> Axis::GetAxe(int n) const { return spLine::New(Axis{o, m_axis_[n]}); }
// std::shared_ptr<spLine> Axis::GetPlaneX() const { return GetAxe(0); }
// std::shared_ptr<spLine> Axis::GetPlaneY() const { return GetAxe(0); }
// std::shared_ptr<spLine> Axis::GetPlaneZ() const { return GetAxe(0); }

}  // namespace geometry{
}  // namespace simpla{