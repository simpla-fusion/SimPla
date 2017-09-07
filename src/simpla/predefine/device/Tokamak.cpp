//
// Created by salmon on 17-7-9.
//
#include "Tokamak.h"
#include <simpla/data/Data.h>
#include <simpla/geometry/occ/GeoObjectOCC.h>

#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <GeomAPI_Interpolate.hxx>
#include <Precision.hxx>
#include <TColgp_HArray1OfPnt.hxx>

namespace simpla {

struct Tokamak::pimpl_s {
    GEqdsk geqdsk;
    Real m_phi0_ = 0, m_phi1_ = TWOPI;
};

Tokamak::Tokamak(std::string const &url) : m_pimpl_(new pimpl_s) { LoadGFile(url); }

Tokamak::~Tokamak() { delete m_pimpl_; }

std::function<Real(point_type const &)> Tokamak::profile(std::string const &attr_name) const {
    std::function<Real(point_type const &)> res = nullptr;

    if (attr_name == "psi") {
        res = [&](point_type const &x) { return m_pimpl_->geqdsk.psi(x); };
    } else if (attr_name == "JT") {
        res = [&](point_type const &x) { return m_pimpl_->geqdsk.JT(x[0], x[1]); };
    } else {
        res = m_pimpl_->geqdsk.GetAttribute(attr_name);
    }

    return res;
};

std::function<Vec3(point_type const &)> Tokamak::B0() const {
    return [&](point_type const &x) { return m_pimpl_->geqdsk.B(x); };
};

void Tokamak::LoadGFile(std::string const &file) { m_pimpl_->geqdsk.load(file); }
std::shared_ptr<geometry::GeoObject> Tokamak::Limiter() const {
    BRepBuilderAPI_MakeWire wireMaker;

    auto num = m_pimpl_->geqdsk.boundary()->data().size();
    Handle(TColgp_HArray1OfPnt) gp_array = new TColgp_HArray1OfPnt(1, static_cast<Standard_Integer>(num));
    auto const &points = m_pimpl_->geqdsk.boundary()->data();
    for (size_type s = 0; s < num - 1; ++s) { gp_array->SetValue(s + 1, gp_Pnt(points[s][0], 0, points[s][1])); }
    GeomAPI_Interpolate sp(gp_array, true, Precision::Confusion());
    sp.Perform();
    wireMaker.Add(BRepBuilderAPI_MakeEdge(sp.Curve()));
    gp_Ax1 axis(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1));
    BRepBuilderAPI_MakeFace myBoundaryFaceProfile(wireMaker.Wire(), true);
    BRepPrimAPI_MakeRevol revol(myBoundaryFaceProfile.Face(), axis);
    return geometry::GeoObjectOCC::New(revol.Shape());
}
std::shared_ptr<geometry::GeoObject> Tokamak::Boundary() const {
    BRepBuilderAPI_MakePolygon polygonMaker;
    for (auto const &p : m_pimpl_->geqdsk.limiter()->data()) { polygonMaker.Add(gp_Pnt(p[0], 0, p[1])); }
    gp_Ax1 axis(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1));
    BRepBuilderAPI_MakeFace myLimterFaceProfile(polygonMaker.Wire());
    BRepPrimAPI_MakeRevol myLimiter(myLimterFaceProfile.Face(), axis);
    return geometry::GeoObjectOCC::New(myLimiter.Shape());
}

}  // namespace simpla {
