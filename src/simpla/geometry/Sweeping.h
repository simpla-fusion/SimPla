//
// Created by salmon on 17-10-24.
//

#ifndef SIMPLA_SWEPTBODY_H
#define SIMPLA_SWEPTBODY_H

#include <simpla/utilities/Constants.h>
#include "GeoObject.h"

namespace simpla {
namespace geometry {
struct Face;
struct Solid;
struct Edge;
struct Wire;
struct gCurve;
std::shared_ptr<GeoObject> MakeRevolution(std::shared_ptr<const GeoEntity> const& geo, Real radius = 0,
                                          vector_type const& Nx = vector_type{1, 0, 0},
                                          vector_type const& Ny = vector_type{0, 1, 0}, Real angle = TWOPI,
                                          Axis const& = Axis{});
std::shared_ptr<GeoObject> MakeRevolution(std::shared_ptr<const GeoEntity> const& geo, Real angle);
std::shared_ptr<GeoObject> MakeRevolution(std::shared_ptr<const GeoEntity> const& geo, Real angle0, Real angle1);
std::shared_ptr<GeoObject> MakePrism(std::shared_ptr<const GeoEntity> const& geo, vector_type const& direction,
                                     Real u = 1, Axis const& = Axis{});

std::shared_ptr<GeoObject> MakePipe(std::shared_ptr<const GeoEntity> const& geo,
                                    std::shared_ptr<const gCurve> const& curve, Axis const& axis = {});
std::shared_ptr<GeoObject> MakeSweep(std::shared_ptr<const GeoEntity> const& face,
                                     std::shared_ptr<const gCurve> const& c, Axis const& axis = {});

std::shared_ptr<Face> MakeSweep(std::shared_ptr<const Edge> const& face, std::shared_ptr<const Edge> const& c,
                                Axis const& axis = {});
std::shared_ptr<Solid> MakeSweep(std::shared_ptr<const Face> const& face, std::shared_ptr<const Edge> const& c,
                                 Axis const& axis = {});

}  // namespace simpla
}  // namespace geometry
#endif  // SIMPLA_SWEPTBODY_H
