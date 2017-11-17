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
Line::Line(point_type const &p0, point_type const &p1) : Curve() {
    auto v = p1 - p0;
    m_length_ = std::sqrt(dot(v, v));
    SetAxis(Axis{p0, (p1 - p0) / m_length_});
};
Line::Line(point_type const &p0, vector_type const &dir, Real l) : Curve(Axis{p0, dir}), m_length_(l) {}

std::shared_ptr<data::DataEntry> Line::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("Length", m_length_);
    return res;
};
void Line::Deserialize(std::shared_ptr<data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    m_length_ = cfg->GetValue("Length", m_length_);
}
bool Line::IsClosed() const { return false; };
point_type Line::xyz(Real u) const { return m_axis_.xyz(u); }
point_type Line::GetStartPoint() const { return m_axis_.o; }
point_type Line::GetEndPoint() const { return m_axis_.o + m_axis_.x * m_length_; }
vector_type Line::GetDirection() const { return m_axis_.x; }

}  // namespace geometry
}  // namespace simpla