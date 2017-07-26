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
class TopoDS_Shape;
namespace simpla {
namespace geometry {

namespace detail {
template <typename TDest, typename TSrc, typename Enable = void>
struct OCCCast {
    static TDest* eval(TSrc const& s) { return nullptr; }
};

gp_Pnt point(point_type const& p0) { return gp_Pnt(p0[0], p0[1], p0[2]); }
gp_Dir dir(vector_type const& p0) { return gp_Dir(p0[0], p0[1], p0[2]); }

template <>
Geom_Curve* OCCCast<Geom_Curve, geometry::Curve>::eval(geometry::Curve const& c) {
    if (c.isA(typeid(Circle))) {
        auto const& l = dynamic_cast<Circle const&>(c);
        return new Geom_Circle(gp_Ax2(point(l.Origin()), dir(l.Normal()), dir(l.XAxis())), l.Radius());
    } else if (c.isA(typeid(Line))) {
        auto const& l = dynamic_cast<Line const&>(c);
        auto p0 = l.Start();
        auto p1 = l.End();
        return new Geom_Line(point(p0), dir(p1 - p0));
    } else {
        UNIMPLEMENTED;
        return nullptr;
    }
};

}  // namespace detail{
template <typename TDest, typename TSrc>
TDest* occ_cast(TSrc const& g) {
    return detail::OCCCast<TDest, TSrc>::eval(g);
}

}  // namespace geometry
}  // namespace simpla
#endif  // SIMPLA_OCCSHAPE_H
