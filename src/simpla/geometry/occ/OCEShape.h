//
// Created by salmon on 17-7-26.
//

#ifndef SIMPLA_OCCSHAPE_H
#define SIMPLA_OCCSHAPE_H


#include "GeoObjectOCE.h"
class TopoDS_Shape;
class Geom_Curve;
class Geom_Surface;
namespace simpla {
namespace geometry {
class Surface;
class Curve;
namespace detail {
template <typename TDest, typename TSrc, typename Enable = void>
struct OCCCast {
    static TDest* eval(TSrc const& s) { return nullptr; }
};

template <>
TopoDS_Shape* OCCCast<TopoDS_Shape, GeoObject>::eval(GeoObject const& g);
template <>
Geom_Curve* OCCCast<Geom_Curve, Curve>::eval(Curve const& c);
template <>
Geom_Surface* OCCCast<Geom_Surface, Surface>::eval(Surface const& c);
}  // namespace detail{
template <typename TDest, typename TSrc>
TDest* occ_cast(TSrc const& g) {
    return detail::OCCCast<TDest, TSrc>::eval(g);
}

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_OCCSHAPE_H
