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
        res = gSweepingSurface::New(geo, gCircle::New(std::sqrt(dot(Nr, Nr))), Nr, cross(Nz, Nr));
    } else if (auto curve = std::dynamic_pointer_cast<const gCurve>(geo)) {
        res = gSweepingBody::New(geo, gCircle::New(std::sqrt(dot(Nr, Nr))), Nr, cross(Nz, Nr));
    }
    return res;
}

std::shared_ptr<gSweeping> gMakePrism(std::shared_ptr<const GeoEntity> const& geo, vector_type const& Nx) {
    return gSweeping::New(geo, gLine::New(), Nx, cross(vector_type{0, 0, 2}, Nx));
}

std::shared_ptr<gSweeping> gMakePipe(std::shared_ptr<const GeoEntity> const& geo,
                                     std::shared_ptr<const gCurve> const& curve) {
    return gSweeping::New(geo, curve, vector_type{1, 0, 0}, vector_type{0, 1, 0});
}

}  // namespace geometry {
}  // namespace simpla {