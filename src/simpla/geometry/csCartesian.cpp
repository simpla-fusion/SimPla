//
// Created by salmon on 17-7-22.
//
#include "csCartesian.h"
#include <memory>
#include "Box.h"
#include "Curve.h"
#include "Line.h"
namespace simpla {
namespace geometry {
csCartesian::csCartesian() = default;
csCartesian::~csCartesian() = default;
std::shared_ptr<simpla::data::DataEntry> csCartesian::Serialize() const { return base_type::Serialize(); }
void csCartesian::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) {}

std::shared_ptr<Curve> csCartesian::GetAxis(point_type const &x0, int dir) const {
    return Line::New(m_axis_.xyz(x0), m_axis_.GetDirection(dir), 1.0);
};
std::shared_ptr<GeoObject> csCartesian::GetBoundingShape(box_type const &uvw) const {
    return Box::New(m_axis_.xyz(std::get<0>(uvw)), m_axis_.xyz(std::get<1>(uvw)));
}
}  // namespace geometry
}  // namespace simpla
