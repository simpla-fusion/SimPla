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
std::shared_ptr<Curve> csCartesian::GetAxis(index_tuple const &idx0, int dir, index_type l) const {
    return GetAxis(m_axis_.uvw(idx0), dir, static_cast<Real>(l));
}
std::shared_ptr<Curve> csCartesian::GetAxis(point_type const &x0, int dir, Real l) const {
    return Line::New(x0, x0 + m_axis_.GetDirection(dir) * l);
};

}  // namespace geometry
}  // namespace simpla
