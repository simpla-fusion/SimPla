//
// Created by salmon on 17-11-22.
//

#include "gSweeping.h"
#include "gCircle.h"
#include "gLine.h"

namespace simpla {
namespace geometry {
std::shared_ptr<GeoEntity> gMakeRevolution(std::shared_ptr<const GeoEntity> const& geo, vector_type const& Nr,
                                           vector_type const& Nz) {
    std::shared_ptr<GeoEntity> res = nullptr;

    if (auto curve = std::dynamic_pointer_cast<const gCurve>(geo)) {
        res = gSweepingSurface::New(curve, gCircle::New(std::sqrt(dot(Nr, Nr))), Nr, cross(Nz, Nr));
    } else if (auto surface = std::dynamic_pointer_cast<const gSurface>(geo)) {
        res = gSweepingBody::New(surface, gCircle::New(std::sqrt(dot(Nr, Nr))), Nr, cross(Nz, Nr));
    }
    return res;
}

std::shared_ptr<GeoEntity> gMakePrism(std::shared_ptr<const GeoEntity> const& geo, vector_type const& Nx) {
    std::shared_ptr<GeoEntity> res = nullptr;

    if (auto curve = std::dynamic_pointer_cast<const gCurve>(geo)) {
        res = gSweepingSurface::New(curve, gLine::New(), Nx, cross(vector_type{0, 0, 2}, Nx));
    } else if (auto surface = std::dynamic_pointer_cast<const gSurface>(geo)) {
        res = gSweepingBody::New(surface, gLine::New(), Nx, cross(vector_type{0, 0, 2}, Nx));
    }
    return res;
}

std::shared_ptr<GeoEntity> gMakePipe(std::shared_ptr<const GeoEntity> const& geo,
                                     std::shared_ptr<const gCurve> const& path) {
    std::shared_ptr<GeoEntity> res = nullptr;
    if (auto curve = std::dynamic_pointer_cast<const gCurve>(geo)) {
        res = gSweepingSurface::New(curve, path, vector_type{1, 0, 0}, vector_type{0, 1, 0});
    } else if (auto surface = std::dynamic_pointer_cast<const gSurface>(geo)) {
        res = gSweepingBody::New(surface, path, vector_type{1, 0, 0}, vector_type{0, 1, 0});
    }
    return res;
}

}  // namespace geometry {
}  // namespace simpla {