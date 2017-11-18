//
// Created by salmon on 17-7-22.
//
#include "csCartesian.h"
#include <memory>
#include "Box.h"
#include "Curve.h"
#include "Edge.h"
#include "Face.h"
#include "Solid.h"
#include "spLine.h"
namespace simpla {
namespace geometry {
csCartesian::csCartesian() = default;
csCartesian::csCartesian(csCartesian const &) = default;
csCartesian::~csCartesian() = default;
std::shared_ptr<simpla::data::DataEntry> csCartesian::Serialize() const { return base_type::Serialize(); }
void csCartesian::Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
}
std::shared_ptr<Edge> csCartesian::GetCoordinateEdge(point_type const &o, int normal, Real u) const {
    return spLine().MakeEdge(m_axis_, std::make_tuple(0.0, u));
};
std::shared_ptr<Face> csCartesian::GetCoordinateFace(point_type const &o, int normal, Real u, Real v) const {
    return spPlane().MakeFace(m_axis_, std::make_tuple(point2d_type{0.0, u}, point2d_type{0.0, v}));
};
std::shared_ptr<Solid> csCartesian::GetCoordinateBox(point_type const &o, Real u, Real v, Real w) const {
    return Solid::New(Box::New(m_axis_.xyz(o)), 0.0, u, 0.0, v, 0.0, w);
}
}  // namespace geometry
}  // namespace simpla
