//
// Created by salmon on 17-10-21.
//
#include "Line.h"
namespace simpla {
namespace geometry {
SP_OBJECT_REGISTER(Line)
std::shared_ptr<data::DataNode> Line::Serialize() const { return base_type::Serialize(); };
void Line::Deserialize(std::shared_ptr<data::DataNode> const &cfg) { base_type::Deserialize(cfg); }

Line::Line() = default;
Line::Line(Line const &) = default;
Line::Line(Axis const &axis, Real alpha0, Real alpha1) : ParametricCurve(axis){};
Line::Line(point_type const &p0, point_type const &p1) : ParametricCurve(Axis{p0, p1 - p0}){};
Line::Line(vector_type const &v) : ParametricCurve(Axis{point_type{0, 0, 0}, v}){};

Line::~Line() = default;
bool Line::IsClosed() const { return false; };
point_type Line::xyz(Real u) const { return m_axis_.xyz(u); }

}  // namespace geometry
}  // namespace simpla