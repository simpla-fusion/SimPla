//
// Created by salmon on 17-7-22.
//
#include "csCartesian.h"
#include "Curve.h"
#include "Line.h"

#include <memory>

namespace simpla {
namespace geometry {
csCartesian::csCartesian() = default;
csCartesian::~csCartesian() = default;
std::shared_ptr<simpla::data::DataNode> csCartesian::Serialize() const { return base_type::Serialize(); }
void csCartesian::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) {}
std::shared_ptr<Curve> csCartesian::GetAxis(index_tuple const &idx0, int dir) const {
    return GetAxis(m_axis_.uvw(idx0), dir);
}
std::shared_ptr<Curve> csCartesian::GetAxis(point_type const &x0, int dir) const {
    return Line::New(x0, m_axis_.GetDirection(dir));
};

}  // namespace geometry
}  // namespace simpla
