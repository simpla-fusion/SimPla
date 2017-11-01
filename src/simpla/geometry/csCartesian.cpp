//
// Created by salmon on 17-7-22.
//
#include "csCartesian.h"
#include "Curve.h"
#include "Line.h"

#include <memory>

namespace simpla {
namespace geometry {
csCartesian::csCartesian() {}
csCartesian::~csCartesian() {}
std::shared_ptr<simpla::data::DataNode> csCartesian::Serialize() const { return base_type::Serialize(); }
void csCartesian::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {}

std::shared_ptr<const GeoObject> csCartesian::GetAxis(point_type const &x0, const point_type &x1) const {
    return Line::New(x0, x1);
}
}  // namespace geometry
}  // namespace simpla
