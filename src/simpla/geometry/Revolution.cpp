//
// Created by salmon on 17-10-24.
//

#include "Revolution.h"
#include "Circle.h"

namespace simpla {
namespace geometry {
Revolution::Revolution() = default;
Revolution::Revolution(Revolution const &other) = default;
Revolution::Revolution(std::shared_ptr<const Surface> const &s, point_type const &origin, vector_type const &axis)
    : Swept(Axis{origin, axis}) {
    //    , Circle::New3(origin, s->GetAxis().o, axis)
    FIXME;
}

Revolution::~Revolution() = default;
void Revolution::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> Revolution::Serialize() const {
    auto res = base_type::Serialize();
    return res;
}

bool Revolution::TestIntersection(box_type const &, Real tolerance) const { return false; }
std::shared_ptr<GeoObject> Revolution::Intersection(std::shared_ptr<const GeoObject> const &, Real tolerance) const {
    return nullptr;
}

/*******************************************************************************************************************/
SP_OBJECT_REGISTER(RevolutionSurface)
RevolutionSurface::RevolutionSurface() = default;
RevolutionSurface::RevolutionSurface(RevolutionSurface const &other) = default;
RevolutionSurface::RevolutionSurface(Axis const &axis, std::shared_ptr<Curve> const &c) : SweptSurface(axis) {}
RevolutionSurface::~RevolutionSurface() = default;
void RevolutionSurface::Deserialize(std::shared_ptr<simpla::data::DataNode> const &cfg) { base_type::Deserialize(cfg); }
std::shared_ptr<simpla::data::DataNode> RevolutionSurface::Serialize() const {
    auto res = base_type::Serialize();
    return res;
}
bool RevolutionSurface::IsClosed() const { return GetBasisCurve()->IsClosed(); };
bool RevolutionSurface::TestIntersection(box_type const &, Real tolerance) const {
    UNIMPLEMENTED;
    return false;
}
std::shared_ptr<GeoObject> RevolutionSurface::Intersection(std::shared_ptr<const GeoObject> const &,
                                                           Real tolerance) const {
    UNIMPLEMENTED;
    return nullptr;
}
}  // namespace geometry{
}  // namespace simpla{