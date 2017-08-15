//
// Created by salmon on 17-8-1.
//
#include "OCCShape.h"
#include <GeomAdaptor_Curve.hxx>
#include <Geom_Circle.hxx>
#include <Geom_Line.hxx>
#include <Geom_Surface.hxx>
#include <Standard_Transient.hxx>
#include <TopoDS_Shape.hxx>
#include "../Curve.h"
#include "../Surface.h"

namespace simpla {
namespace geometry {

namespace detail {
gp_Pnt point(point_type const& p0) { return gp_Pnt{p0[0], p0[1], p0[2]}; }
gp_Dir dir(vector_type const& p0) { return gp_Dir{p0[0], p0[1], p0[2]}; }
template <>
TopoDS_Shape* OCCCast<TopoDS_Shape, GeoObject>::eval(GeoObject const& g) {
    auto* res = new TopoDS_Shape;
    if (dynamic_cast<GeoObjectOCC const*>(&g) != nullptr) {
        *res = dynamic_cast<GeoObjectOCC const&>(g).GetShape();
    } else {
        auto p = GeoObjectOCC::New(g);
        *res = p->GetShape();
    }
    return res;
}
template <>
Geom_Curve* OCCCast<Geom_Curve, Curve>::eval(Curve const& c) {
    Geom_Curve* res = nullptr;
    if (dynamic_cast<Circle const*>(&c) != nullptr) {
        auto const& l = dynamic_cast<Circle const&>(c);
        res = new Geom_Circle(gp_Ax2(point(l.Origin()), dir(l.Normal()), dir(l.XAxis())), l.Radius());
    } else if (dynamic_cast<Line const*>(&c) != nullptr) {
        auto const& l = dynamic_cast<Line const&>(c);
        res = new Geom_Line(point(l.Origin()), dir(l.Direction()));
    } else {
        UNIMPLEMENTED;
    }
    return res;
};
}

}  // namespace geometry
}  // namespace simpla