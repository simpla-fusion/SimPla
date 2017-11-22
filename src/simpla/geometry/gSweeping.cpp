//
// Created by salmon on 17-11-22.
//

#include "gSweeping.h"
#include "gCircle.h"
#include "gLine.h"

namespace simpla {
namespace geometry {
std::shared_ptr<gSweeping> MakeRevolution(std::shared_ptr<const GeoEntity> const& geo, vector_type const& Nr,
                                          vector_type const& Nz) {
    return gSweeping::New(geo, gCircle::New(std::sqrt(dot(Nr, Nr))), Nr, cross(Nz, Nr));
}

std::shared_ptr<gSweeping> MakePrism(std::shared_ptr<const GeoEntity> const& geo, vector_type const& Nx) {
    return gSweeping::New(geo, gLine::New(), Nx, cross(vector_type{0, 0, 2}, Nx));
}

std::shared_ptr<gSweeping> MakePipe(std::shared_ptr<const GeoEntity> const& geo,
                                    std::shared_ptr<const gCurve> const& curve) {
    return gSweeping::New(geo, curve, vector_type{1, 0, 0}, vector_type{0, 1, 0});
}

}  // namespace geometry {
}  // namespace simpla {