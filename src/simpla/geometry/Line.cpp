//
// Created by salmon on 17-10-21.
//
#include "Line.h"
namespace simpla {
namespace geometry {
SP_GEO_OBJECT_REGISTER(Line)

Line::Line() = default;
Line::Line(Line const &) = default;
Line::~Line() = default;
Line::Line(Axis const &axis) : Curve(axis){};
Line::Line(point_type const &p0, point_type const &p1) : Curve(Axis{p0, p1 - p0}){};
Line::Line(vector_type const &v) : Curve(Axis{point_type{0, 0, 0}, v}){};
std::shared_ptr<data::DataNode> Line::Serialize() const { return base_type::Serialize(); };
void Line::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
bool Line::IsClosed() const { return false; };
point_type Line::xyz(Real u) const { return m_axis_.xyz(u); }
point_type Line::GetStartPoint() const { return m_axis_.o; }
point_type Line::GetEndPoint() const { return m_axis_.o + m_axis_.x; }

}  // namespace geometry
}  // namespace simpla