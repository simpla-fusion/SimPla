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
std::shared_ptr<GeoObject> MakeRevolution(std::shared_ptr<const GeoEntity> const& geo, Axis const& r_axis,
                                          box_type const& range, Axis const& g_axis) {
    return GeoObjectHandle::New(g_axis, gSweeping::New(geo, gCircle::New(), r_axis), range);
};
std::shared_ptr<GeoObject> MakeRevolution(std::shared_ptr<const GeoEntity> const& geo, Real angle0, Real angle1,
                                          Real u0, Real u1, Real v0, Real v1, Real radius, vector_type const& Nx,
                                          vector_type const& Ny, Axis const& g_axis) {
    Axis r_axis;
    r_axis.SetOrigin(point_type{radius, 0, 0});
    r_axis.SetAxis(0, Nx);
    r_axis.SetAxis(1, Ny);
    r_axis.SetAxis(2, cross(Nx, Ny));

    return MakeRevolution(geo, r_axis, box_type{{u0, v0, angle0}, {u1, v1, angle1}}, g_axis);
};
std::shared_ptr<GeoObject> MakeRevolution(std::shared_ptr<const GeoEntity> const& geo, Real angle, Axis const& axis) {
    return MakeRevolution(geo, 0, angle, 0, 1, 0, 1, 0, vector_type{1, 0, 0}, vector_type{0, 0, 1}, axis);
}
std::shared_ptr<GeoObject> MakeRevolution(std::shared_ptr<const GeoEntity> const& geo, Real angle0, Real angle1,
                                          Axis const& axis) {
    return MakeRevolution(geo, angle0, angle1, 0, 1, 0, 1, 0, vector_type{1, 0, 0}, vector_type{0, 0, 1}, axis);
}

std::shared_ptr<GeoObject> MakeRevolution(std::shared_ptr<const GeoEntity> const& geo, Axis const& r_axis, Real radius,
                                          Axis const& g_axis) {
    return GeoObjectHandle::New(g_axis, gSweeping::New(geo, gCircle::New(radius), r_axis));
}

std::shared_ptr<GeoObject> MakePrism(std::shared_ptr<const GeoEntity> const& geo, vector_type const& direction, Real u,
                                     Axis const& g_axis) {
    return GeoObjectHandle::New(g_axis, gSweeping::New(geo, gLine::New(direction)));
}

std::shared_ptr<GeoObject> MakePipe(std::shared_ptr<const GeoEntity> const& geo,
                                    std::shared_ptr<const gCurve> const& path, Axis const& r_axis, Axis const& g_axis) {
    return GeoObjectHandle::New(g_axis, gSweeping::New(geo, path, r_axis));
}
std::shared_ptr<GeoObject> MakeSweep(std::shared_ptr<const GeoEntity> const& geo,
                                     std::shared_ptr<const gCurve> const& path, Axis const& r_axis,
                                     Axis const& g_axis) {
    return GeoObjectHandle::New(g_axis, gSweeping::New(geo, path, r_axis));
}
std::shared_ptr<Face> MakeSweep(std::shared_ptr<const Edge> const& face, std::shared_ptr<const Edge> const& c,
                                Axis const& g_axis) {
    UNIMPLEMENTED;
    return nullptr;
}
std::shared_ptr<Solid> MakeSweep(std::shared_ptr<const Face> const& face, std::shared_ptr<const Edge> const& c,
                                 Axis const& g_axis) {
    UNIMPLEMENTED;
    return nullptr;
}
}  // namespace geometry{
}  // namespace simpla{