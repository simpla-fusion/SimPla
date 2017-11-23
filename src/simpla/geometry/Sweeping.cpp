//
// Created by salmon on 17-10-24.
//
#include "Sweeping.h"
#include "GeoObject.h"
#include "gCircle.h"
#include "gLine.h"
#include "gSweeping.h"

namespace simpla {
namespace geometry {

std::shared_ptr<GeoObject> MakeRevolution(std::shared_ptr<const GeoEntity> const& geo) {
    return GeoObjectHandle::New(gSweeping::New(geo, gCircle::New(0), Axis{}));
}
std::shared_ptr<GeoObject> MakeRevolution(std::shared_ptr<const GeoEntity> const& geo, vector_type const& Nz,
                                          Real radius) {
    return GeoObjectHandle::New(gSweeping::New(geo, gCircle::New(radius), Axis{}));
};
std::shared_ptr<GeoObject> MakeRevolution(std::shared_ptr<const GeoEntity> const& geo, Axis const& r_axis,
                                          Real radius) {
    return GeoObjectHandle::New(gSweeping::New(geo, gCircle::New(radius), r_axis));
}

std::shared_ptr<GeoObject> MakePrism(std::shared_ptr<const GeoEntity> const& geo, vector_type const& direction,
                                     Real u) {
    return GeoObjectHandle::New(gSweeping::New(geo, gLine::New(direction)));
}

std::shared_ptr<GeoObject> MakePipe(std::shared_ptr<const GeoEntity> const& geo,
                                    std::shared_ptr<const gCurve> const& path, Axis const& r_axis) {
    return GeoObjectHandle::New(gSweeping::New(geo, path, r_axis));
}
std::shared_ptr<GeoObject> MakeSweep(std::shared_ptr<const GeoEntity> const& geo,
                                     std::shared_ptr<const gCurve> const& path, Axis const& r_axis) {
    return GeoObjectHandle::New(gSweeping::New(geo, path, r_axis));
}
std::shared_ptr<Face> MakeSweep(std::shared_ptr<const Edge> const& face, std::shared_ptr<const Edge> const& c) {
    UNIMPLEMENTED;
    return nullptr;
}
std::shared_ptr<Solid> MakeSweep(std::shared_ptr<const Face> const& face, std::shared_ptr<const Edge> const& c) {
    UNIMPLEMENTED;
    return nullptr;
}
}  // namespace geometry{
}  // namespace simpla{