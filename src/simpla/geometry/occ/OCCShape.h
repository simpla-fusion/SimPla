//
// Created by salmon on 17-7-26.
//

#ifndef SIMPLA_OCCSHAPE_H
#define SIMPLA_OCCSHAPE_H

#include <GeomAdaptor_Curve.hxx>
#include <Geom_Circle.hxx>
#include <Geom_Line.hxx>
#include <Standard_Transient.hxx>
#include <TopoDS_Shape.hxx>
#include "../Curve.h"
#include "GeoObjectOCC.h"
class TopoDS_Shape;
namespace simpla {
namespace geometry {

namespace detail {
template <typename TDest, typename TSrc, typename Enable = void>
struct OCCCast {
    static TDest* eval(TSrc const& s) { return nullptr; }
};

gp_Pnt point(point_type const& p0) { return gp_Pnt{p0[0], p0[1], p0[2]}; }
gp_Dir dir(vector_type const& p0) { return gp_Dir{p0[0], p0[1], p0[2]}; }

template <>
TopoDS_Shape* OCCCast<TopoDS_Shape, GeoObject>::eval(GeoObject const& g) {
    auto* res = new TopoDS_Shape;
    if (g.isA(typeid(GeoObjectOCC))) {
        *res = dynamic_cast<GeoObjectOCC const&>(g).GetShape();
    } else {
        *res = GeoObjectOCC(g).GetShape();
    }
    return res;
}
template <>
Geom_Curve* OCCCast<Geom_Curve, Curve>::eval(Curve const& c) {
    Geom_Curve* res = nullptr;
    if (c.isA(typeid(Circle))) {
        auto const& l = dynamic_cast<Circle const&>(c);
        res = new Geom_Circle(gp_Ax2(point(l.Origin()), dir(l.Normal()), dir(l.XAxis())), l.Radius());
    } else if (c.isA(typeid(Line))) {
        auto const& l = dynamic_cast<Line const&>(c);
        res = new Geom_Line(point(l.Origin()), dir(l.Direction()));
    } else {
        UNIMPLEMENTED;
    }
    return res;
};

}  // namespace detail{
template <typename TDest, typename TSrc>
TDest* occ_cast(TSrc const& g) {
    return detail::OCCCast<TDest, TSrc>::eval(g);
}

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_OCCSHAPE_H
