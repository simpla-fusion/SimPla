//
// Created by salmon on 17-10-24.
//

#include "Revolution.h"
#include "Circle.h"

namespace simpla {
namespace geometry {
Revolution::Revolution() = default;
Revolution::Revolution(Revolution const &other) = default;
Revolution::Revolution(std::shared_ptr<const Surface> const &s, point_type const &origin, vector_type const &axis,
                       Real phi0, Real phi1)
    : SweptBody(s->Moved(s->GetAxis().o - origin), Circle::New3(origin, s->GetAxis().o, axis)) {
    FIXME;
}

Revolution::~Revolution() = default;
void Revolution::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> Revolution::Serialize() const {
    auto res = base_type::Serialize();
    return res;
}
point_type Revolution::Value(point_type const &uvw) const {
    auto p = m_basis_surface_->Value(uvw[0], uvw[1]);
    auto cosW = std::cos(uvw[2]);
    auto sinW = std::sin(uvw[2]);
    return point_type{p[0] * cosW - p[1] * sinW, p[0] * sinW + p[1] * cosW, p[2]};
}
bool Revolution::TestIntersection(box_type const &) const { return false; }
std::shared_ptr<GeoObject> Revolution::Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const {
    return nullptr;
}

/*******************************************************************************************************************/
SP_OBJECT_REGISTER(RevolutionSurface)

void RevolutionSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> RevolutionSurface::Serialize() const {
    auto res = base_type::Serialize();
    return res;
}
bool RevolutionSurface::TestIntersection(box_type const &) const { return 0; }
std::shared_ptr<GeoObject> RevolutionSurface::Intersection(std::shared_ptr<const GeoObject> const &,
                                                           Real tolerance) const {
    return nullptr;
}
}  // namespace geometry{
}  // namespace simpla{