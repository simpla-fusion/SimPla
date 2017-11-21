//
// Created by salmon on 17-11-1.
//

#include "GeoEngineOCE.h"
#include "../GeoObject.h"

#include <simpla/algebra/nTuple.ext.h>

#include <simpla/geometry/Chart.h>
#include <simpla/geometry/gCone.h>
#include <simpla/geometry/gCylinder.h>
#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeFace.hxx>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepBuilderAPI_MakeWire.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepIntCurveSurface_Inter.hxx>
#include <BRepOffsetAPI_MakePipe.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeCylinder.hxx>
#include <BRepPrimAPI_MakePrism.hxx>
#include <BRepPrimAPI_MakeRevol.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRepPrimAPI_MakeTorus.hxx>
#include <BRepPrimAPI_MakeWedge.hxx>
#include <Bnd_Box.hxx>
#include <GeomAPI_Interpolate.hxx>
#include <GeomAdaptor_Curve.hxx>
#include <Geom_Circle.hxx>
#include <Geom_ConicalSurface.hxx>
#include <Geom_Curve.hxx>
#include <Geom_CylindricalSurface.hxx>
#include <Geom_Ellipse.hxx>
#include <Geom_Hyperbola.hxx>
#include <Geom_Line.hxx>
#include <Geom_Parabola.hxx>
#include <Geom_Plane.hxx>
#include <Geom_SphericalSurface.hxx>
#include <Geom_Surface.hxx>
#include <Handle_TDocStd_Document.hxx>
#include <Handle_XCAFApp_Application.hxx>
#include <Handle_XCAFDoc_ShapeTool.hxx>
#include <IGESCAFControl_Writer.hxx>
#include <Interface_Static.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <STEPControl_Reader.hxx>
#include <STEPControl_StepModelType.hxx>
#include <Standard_Transient.hxx>
#include <StlAPI_Reader.hxx>
#include <StlAPI_Writer.hxx>
#include <TColStd_HSequenceOfTransient.hxx>
#include <TColgp_HArray1OfPnt.hxx>
#include <TDataStd_Name.hxx>
#include <TDocStd_Document.hxx>
#include <TopoDS_Shape.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <gp_Pln.hxx>
#include <gp_Quaternion.hxx>
#include "../Box.h"
#include "../Edge.h"
#include "../Face.h"
#include "../GeoEntity.h"
#include "../GeoObject.h"
#include "../IntersectionCurveSurface.h"
#include "../Revolution.h"
#include "../gCircle.h"
#include "../gEllipse.h"
#include "../gHyperbola.h"
#include "../gLine.h"
#include "../gParabola.h"
#include "../gPlane.h"
#include "../gPolygon.h"
#include "../gSphere.h"
#include "../gTorus.h"
#include "../gWedge.h"
#include "simpla/geometry/gBody.h"
#include "simpla/geometry/gBoundedCurve.h"
#include "simpla/geometry/gCurve.h"
#include "simpla/geometry/gSurface.h"

namespace simpla {
namespace geometry {
struct GeoObjectOCE;

template <typename TDest, typename TSrc>
struct OCEShapeCast {
    static std::shared_ptr<TDest> eval(std::shared_ptr<const TSrc> const &s);
};
template <typename TDest, typename TSrc>
std::shared_ptr<TDest> oce_cast(std::shared_ptr<const TSrc> const &s);

gp_Pnt make_point(point_type const &p0) { return gp_Pnt{p0[0], p0[1], p0[2]}; }
gp_Dir make_dir(vector_type const &p0) { return gp_Dir{p0[0], p0[1], p0[2]}; }
gp_Ax2 make_axis(point_type const &origin, vector_type const &z, vector_type const &x) {
    return gp_Ax2{make_point(origin), make_dir(z), make_dir(x)};
}
gp_Ax2 make_axis(Axis const &axis) { return gp_Ax2{make_point(axis.o), make_dir(axis.z), make_dir(axis.x)}; }
gp_Ax3 make_axis3(Axis const &axis) { return gp_Ax2{make_point(axis.o), make_dir(axis.z), make_dir(axis.x)}; }
gp_Ax1 make_axis1(Axis const &axis) { return gp_Ax1{make_point(axis.o), make_dir(axis.z)}; }

template <>
std::shared_ptr<TopoDS_Shape> OCEShapeCast<TopoDS_Shape, gSurface>::eval(std::shared_ptr<const gSurface> const &g) {
    std::shared_ptr<TopoDS_Shape> res = nullptr;
    UNIMPLEMENTED;
    return res;
};

struct GeoObjectOCE : public GeoObject {
    SP_GEO_OBJECT_HEAD(GeoObject, GeoObjectOCE)
    void Deserialize(std::shared_ptr<const simpla::data::DataEntry> const &cfg) override;
    std::shared_ptr<simpla::data::DataEntry> Serialize() const override;

   public:
    explicit GeoObjectOCE(TopoDS_Shape const &shape);
    explicit GeoObjectOCE(std::shared_ptr<TopoDS_Shape> const &shape);
    explicit GeoObjectOCE(GeoObject const &g);
    explicit GeoObjectOCE(std::shared_ptr<const GeoObject> const &g);

    void DoUpdate();

    std::shared_ptr<TopoDS_Shape> GetShape() const;
    Bnd_Box const &GetOCCBoundingBox() const;

    std::shared_ptr<GeoObject> GetBoundary() const override;
    box_type GetBoundingBox() const override;
    bool CheckIntersection(point_type const &x, Real tolerance) const override;
    bool CheckIntersection(box_type const &, Real tolerance) const override;
    std::shared_ptr<GeoObject> GetUnion(std::shared_ptr<const GeoObject> const &g, Real tolerance) const override;
    std::shared_ptr<GeoObject> GetDifference(std::shared_ptr<const GeoObject> const &g, Real tolerance) const override;
    std::shared_ptr<GeoObject> GetIntersection(std::shared_ptr<const GeoObject> const &g,
                                               Real tolerance) const override;

   private:
    Real m_measure_ = SP_SNaN;
    std::shared_ptr<TopoDS_Shape> m_occ_shape_ = nullptr;
    box_type m_bounding_box_{{0, 0, 0}, {0, 0, 0}};

    Bnd_Box m_occ_box_;
};
bool GeoObjectOCE::_is_registered = simpla::Factory<GeoObject>::RegisterCreator<GeoObjectOCE>("oce");

template <>
std::shared_ptr<TopoDS_Shape> OCEShapeCast<TopoDS_Shape, Solid>::eval(std::shared_ptr<const Solid> const &g) {
    auto const &axis = make_axis(g->GetAxis());
    auto body = g->GetBody();
    point_type p_min, p_max;
    std::tie(p_min, p_max) = g->GetParameterRange();
    std::shared_ptr<TopoDS_Shape> res = nullptr;
    //    if (auto plane = std::dynamic_pointer_cast<const gPlane>(g)) {
    //        res = std::make_shared<TopoDS_Face>(BRepPrimAPI_MakeBox(make_axis(box->GetAxis()), box->GetExtents()[0],
    //                                                                box->GetExtents()[1], box->GetExtents()[2])
    //                                                .GeoEntity());
    //    } else
    if (auto box = std::dynamic_pointer_cast<const Box>(body)) {
        res = std::make_shared<TopoDS_Shape>(
            BRepPrimAPI_MakeBox(axis, box->GetExtents()[0], box->GetExtents()[1], box->GetExtents()[2]).Shape());
    } else if (auto wedge = std::dynamic_pointer_cast<const gWedge>(g)) {
        res = std::make_shared<TopoDS_Shape>(BRepPrimAPI_MakeWedge(axis, wedge->GetExtents()[0], wedge->GetExtents()[1],
                                                                   wedge->GetExtents()[2], wedge->GetLTX())
                                                 .Shape());
    } else if (auto sphere = std::dynamic_pointer_cast<const gSphere>(body)) {
        res = std::make_shared<TopoDS_Shape>(BRepPrimAPI_MakeSphere(axis, sphere->GetRadius()).Shape());
    } else if (auto cylinder = std::dynamic_pointer_cast<const gCylinder>(body)) {
        res = std::make_shared<TopoDS_Shape>(
            BRepPrimAPI_MakeCylinder(axis, cylinder->GetRadius(), cylinder->GetHeight()).Shape());
    } else if (auto torus = std::dynamic_pointer_cast<const gTorus>(g)) {
        res = std::make_shared<TopoDS_Shape>(
            BRepPrimAPI_MakeTorus(axis, torus->GetMajorRadius(), torus->GetMinorRadius()).Shape());
    }

    //    else if (auto revolution = std::dynamic_pointer_cast<const Revolution>(g)) {
    //        if (auto basis_shape = oce_cast<TopoDS_Shape>(revolution->GetBasisObject())) {
    //            res = std::make_shared<TopoDS_Shape>(
    //                BRepPrimAPI_MakeRevol(*basis_shape, make_axis1(revolution->GetAxis()),
    //                revolution->GetAngle()).Shape());
    //        }
    //    } else if (auto sweep = std::dynamic_pointer_cast<const Sweep>(g)) {
    //        if (auto line = std::dynamic_pointer_cast<const gLine>(sweep->GetCurve())) {
    //            auto basis_shape = oce_cast<TopoDS_Shape>(revolution->GetBasisObject());
    //            res = std::make_shared<TopoDS_Shape>(
    //                BRepPrimAPI_MakePrism(*basis_shape, make_dir(line->GetEndPoint() -
    //                line->GetStartPoint())).Shape());
    //        } else {
    //            res =
    //                std::make_shared<TopoDS_Shape>(BRepOffsetAPI_MakePipe(*oce_cast<TopoDS_Wire>(sweep->GetCurve()),
    //                                                                      *oce_cast<TopoDS_Shape>(sweep->GetBasisObject()))
    //                                                   .Shape());
    //        }
    //
    //    }

    else {
        UNIMPLEMENTED;
    }
    return res;
};

Handle(Geom_Curve) make_geo_curve(Axis const &g_axis, std::shared_ptr<const gCurve> const &curve) {
    auto axis = make_axis(g_axis);
    Handle(Geom_Curve) c;
    if (auto line = std::dynamic_pointer_cast<const gLine>(curve)) {
        c = new Geom_Line(make_point(g_axis.o), make_dir(g_axis.x));
    } else if (auto circle = std::dynamic_pointer_cast<const gCircle>(curve)) {
        c = new Geom_Circle(axis, circle->GetRadius());
    } else if (auto ellipse = std::dynamic_pointer_cast<const gEllipse>(curve)) {
        c = new Geom_Ellipse(axis, ellipse->GetMajorRadius(), ellipse->GetMinorRadius());
    } else if (auto hyperbola = std::dynamic_pointer_cast<const gHyperbola>(curve)) {
        c = new Geom_Hyperbola(axis, hyperbola->GetMajorRadius(), hyperbola->GetMinorRadius());
    } else if (auto parabola = std::dynamic_pointer_cast<const gParabola>(curve)) {
        c = new Geom_Parabola(axis, parabola->GetFocal());
    } else {
        UNIMPLEMENTED;
    }
    return c;
};
Handle(Geom_Curve) make_geo_curve(std::shared_ptr<const Edge> const &edge) {
    return make_geo_curve(edge->GetAxis(), edge->GetCurve());
}
Handle(Geom_Surface) make_geo_surface(Axis const &g_axis, std::shared_ptr<const gSurface> const &surface) {
    auto const &axis = make_axis(g_axis);
    Handle(Geom_Surface) res;
    if (auto plane = std::dynamic_pointer_cast<const gPlane>(surface)) {
        res = new Geom_Plane(make_point(g_axis.o), make_dir(g_axis.z));
    } else if (auto cylinder = std::dynamic_pointer_cast<const gCylinder>(surface)) {
        res = new Geom_CylindricalSurface(make_axis3(g_axis), cylinder->GetRadius());
    } else if (auto sphere = std::dynamic_pointer_cast<const gSphere>(surface)) {
        res = new Geom_SphericalSurface(make_axis3(g_axis), sphere->GetRadius());
    } else if (auto cone = std::dynamic_pointer_cast<const gCone>(surface)) {
        res = new Geom_ConicalSurface(make_axis3(g_axis), cone->GetAngle(), cone->GetRadius());
    } else {
        UNIMPLEMENTED;
    }
    return res;
}
Handle(Geom_Surface) make_geo_surface(std::shared_ptr<const Face> const &surface) {
    return make_geo_surface(surface->GetAxis(), surface->GetSurface());
}
template <>
std::shared_ptr<TopoDS_Edge> OCEShapeCast<TopoDS_Edge, Edge>::eval(std::shared_ptr<const Edge> const &g) {
    std::shared_ptr<TopoDS_Edge> res = nullptr;
    if (auto bc = std::dynamic_pointer_cast<const gBoundedCurve>(g)) {
        BRepBuilderAPI_MakeWire wireMaker;
        auto num = bc->size() - 2;
        Handle(TColgp_HArray1OfPnt) gp_array = new TColgp_HArray1OfPnt(1, static_cast<Standard_Integer>(num));
        for (size_type s = 0; s < num - 1; ++s) { gp_array->SetValue(s + 1, make_point(bc->GetPoint(s))); }
        GeomAPI_Interpolate sp(gp_array, true, Precision::Confusion());
        sp.Perform();
        res = std::make_shared<TopoDS_Edge>(BRepBuilderAPI_MakeEdge(sp.Curve()));
    } else {
        res = std::make_shared<TopoDS_Edge>(BRepBuilderAPI_MakeEdge(make_geo_curve(g->GetAxis(), g->GetCurve())));
    }
    return res;
}
template <>
std::shared_ptr<TopoDS_Wire> OCEShapeCast<TopoDS_Wire, gCurve>::eval(std::shared_ptr<const gCurve> const &g) {
    std::shared_ptr<TopoDS_Wire> res = nullptr;
    if (auto polygon = std::dynamic_pointer_cast<const gPolygon>(g)) {
        BRepBuilderAPI_MakePolygon oce_polygon;
        for (size_type s = 0, se = polygon->size(); s < se; ++s) { oce_polygon.Add(make_point(polygon->GetPoint(s))); }
        oce_polygon.Build();
        res = std::make_shared<TopoDS_Wire>(oce_polygon.Wire());
    } else {
        UNIMPLEMENTED;
    }
    return res;
};
template <>
std::shared_ptr<TopoDS_Face> OCEShapeCast<TopoDS_Face, Face>::eval(std::shared_ptr<const Face> const &face) {
    std::shared_ptr<TopoDS_Face> res = nullptr;
    point2d_type p_min, p_max;
    std::tie(p_min, p_max) = face->GetParameterRange();
    if (auto plane = std::dynamic_pointer_cast<const gPlane>(face->GetSurface())) {
        gp_Pln g_plane{make_point(face->GetAxis().o), make_dir(face->GetAxis().z)};
        res = std::make_shared<TopoDS_Face>(
            BRepBuilderAPI_MakeFace(g_plane, p_min[0], p_max[0], p_min[1], p_max[1]).Face());
    } else {
        UNIMPLEMENTED;
    }
    return res;
};
template <>
std::shared_ptr<TopoDS_Solid> OCEShapeCast<TopoDS_Solid, Solid>::eval(std::shared_ptr<const Solid> const &solid) {
    std::shared_ptr<TopoDS_Solid> res = nullptr;
    //    point2d_type p_min, p_max;
    //    std::tie(p_min, p_max) = face->GetParameterRange();
    //    if (auto plane = std::dynamic_pointer_cast<const gPlane>(face->GetSurface())) {
    //        gp_Pln g_plane{make_point(face->GetAxis().o), make_dir(face->GetAxis().z)};
    //        res = std::make_shared<TopoDS_Face>(
    //            BRepBuilderAPI_MakeFace(g_plane, p_min[0], p_max[0], p_min[1], p_max[1]).Face());
    //    } else {
    //        UNIMPLEMENTED;
    //    }
    return res;
};
template <>
std::shared_ptr<TopoDS_Shape> OCEShapeCast<TopoDS_Shape, GeoObject>::eval(std::shared_ptr<const GeoObject> const &g) {
    std::shared_ptr<TopoDS_Shape> res = nullptr;
    if (auto oce = std::dynamic_pointer_cast<GeoObjectOCE const>(g)) {
        res = oce->GetShape();
    } else if (auto c = std::dynamic_pointer_cast<Edge const>(g)) {
        res = oce_cast<TopoDS_Edge>(c);
    } else if (auto f = std::dynamic_pointer_cast<Face const>(g)) {
        res = oce_cast<TopoDS_Face>(f);
    } else if (auto s = std::dynamic_pointer_cast<Solid const>(g)) {
        res = oce_cast<TopoDS_Solid>(s);
    }
    //    else if (auto s = std::dynamic_pointer_cast<gSurface const>(g)) {
    //        res = oce_cast<TopoDS_Face>(s);
    //    }
    //    else if (auto b = std::dynamic_pointer_cast<gBody const>(g)) {
    //        res = oce_cast<TopoDS_Solid>(b);
    //    }
    else {
        LOGGER << *g;
        UNIMPLEMENTED;
    }
    return res;
};

template <typename TDest, typename TSrc>
std::shared_ptr<TDest> oce_cast(std::shared_ptr<const TSrc> const &s) {
    return OCEShapeCast<TDest, TSrc>::eval(s);
};

GeoObjectOCE::GeoObjectOCE() = default;
GeoObjectOCE::GeoObjectOCE(GeoObjectOCE const &) = default;
GeoObjectOCE::~GeoObjectOCE() = default;

GeoObjectOCE::GeoObjectOCE(TopoDS_Shape const &shape) : m_occ_shape_(std::make_shared<TopoDS_Shape>(shape)) {
    DoUpdate();
}
GeoObjectOCE::GeoObjectOCE(std::shared_ptr<TopoDS_Shape> const &shape) : m_occ_shape_(shape) { DoUpdate(); }
GeoObjectOCE::GeoObjectOCE(std::shared_ptr<const GeoObject> const &g)
    : GeoObject(*g), m_occ_shape_(oce_cast<TopoDS_Shape>(g)) {
    DoUpdate();
}
GeoObjectOCE::GeoObjectOCE(GeoObject const &g) : m_occ_shape_(oce_cast<TopoDS_Shape>(g.shared_from_this())) {
    DoUpdate();
};

std::shared_ptr<TopoDS_Shape> GeoObjectOCE::GetShape() const { return m_occ_shape_; }
Bnd_Box const &GeoObjectOCE::GetOCCBoundingBox() const { return m_occ_box_; }

std::shared_ptr<TopoDS_Shape> ReadSTEP(std::string const &file_name) {
    STEPControl_Reader reader;
    IFSelect_ReturnStatus stat = reader.ReadFile(file_name.c_str());
    ASSERT(stat == IFSelect_RetDone);  // ExcMessage("Error in reading file!"));
    Standard_Boolean failsonly = Standard_False;
    IFSelect_PrintCount mode = IFSelect_ItemsByEntity;
    reader.PrintCheckLoad(failsonly, mode);
    Standard_Integer nRoots = reader.TransferRoots();
    ASSERT(nRoots > 0);  //, 262 ExcMessage("Read nothing from file."));
    VERBOSE << "STEP Object is loaded from " << file_name << "[" << nRoots << "]" << std::endl;
    return std::make_shared<TopoDS_Shape>(reader.OneShape());
}
std::shared_ptr<TopoDS_Shape> ReadSTL(std::string const &file_name) {
    StlAPI_Reader reader;
    TopoDS_Shape shape;
    reader.Read(shape, file_name.c_str());
    return std::make_shared<TopoDS_Shape>(shape);
}

std::shared_ptr<TopoDS_Shape> TransformShape(std::shared_ptr<const TopoDS_Shape> const &shape, Real scale,
                                             point_type const &location, nTuple<Real, 4> const &rotate) {
    auto res = std::make_shared<TopoDS_Shape>();
    *res = *shape;
    // Handle STEP Scale here.
    gp_Pnt origin{location[0], location[1], location[2]};
    gp_Quaternion rot_v{rotate[0], rotate[1], rotate[2], rotate[3]};
    gp_Trsf transf;
    transf.SetScale(origin, scale);
    //    transf.SetRotation(rot_v);
    BRepBuilderAPI_Transform trans(*res, transf);

    return res;
}

std::shared_ptr<TopoDS_Shape> LoadOCEShape(std::string const &file_name, std::string const &obj_name) {
    std::shared_ptr<TopoDS_Shape> res = nullptr;
    std::string ext = file_name.substr(file_name.rfind('.') + 1);
    if (ext == "step" || ext == "stp") {
        res = ReadSTEP(file_name);
    } else if (ext == "stl") {
        res = ReadSTL(file_name);
    }
    return res;
};
int SaveOCEShape(std::shared_ptr<const TopoDS_Shape> const &shape, std::string const &file_name,
                 std::string const &obj_name) {
    if (shape == nullptr) { return SP_FAILED; }
    std::string ext = file_name.substr(file_name.rfind('.'));
    if (ext.empty()) { ext = ".stp"; }

    if (ext == ".step" || ext == ".stp") {
        Handle(TDocStd_Document) aDoc;
        Handle(XCAFApp_Application) anApp = XCAFApp_Application::GetApplication();
        anApp->NewDocument("MDTV-XCAF", aDoc);
        Handle(XCAFDoc_ShapeTool) myShapeTool = XCAFDoc_DocumentTool::ShapeTool(aDoc->Main());
        TDF_Label aLabel1 = myShapeTool->NewShape();
        Handle(TDataStd_Name) NameAttrib1 = new TDataStd_Name();
        NameAttrib1->Set(obj_name.c_str());
        aLabel1.AddAttribute(NameAttrib1);
        myShapeTool->SetShape(aLabel1, *shape);
        STEPCAFControl_Writer().Perform(aDoc, file_name.c_str());
    } else if (ext == ".stl") {
        UNIMPLEMENTED;
    } else {
        UNIMPLEMENTED;
    }

    return SP_SUCCESS;
};

std::shared_ptr<data::DataEntry> GeoObjectOCE::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("HashCode", m_occ_shape_->HashCode(std::numeric_limits<int>::max()));
    return res;
};
void GeoObjectOCE::Deserialize(std::shared_ptr<const data::DataEntry> const &cfg) {
    base_type::Deserialize(cfg);
    auto tdb = std::dynamic_pointer_cast<const data::DataEntry>(cfg);
    if (tdb != nullptr) {
        m_occ_shape_ = TransformShape(
            LoadOCEShape(tdb->GetValue<std::string>("File", ""), ""),
            tdb->GetValue<Real>("Scale", 1.0e-3),            // default length unit is "m", STEP length unit is "mm"
            tdb->GetValue("Location", point_type{0, 0, 0}),  //
            tdb->GetValue("Rotation", nTuple<Real, 4>{0, 0, 0, 0}));
    }
    DoUpdate();
    VERBOSE << " [ Bounding Box :" << m_bounding_box_ << "]" << std::endl;
};

void GeoObjectOCE::DoUpdate() {
    ASSERT(m_occ_shape_ != nullptr);
    BRepBndLib::Add(*m_occ_shape_, m_occ_box_);
    m_occ_box_.Get(std::get<0>(m_bounding_box_)[0], std::get<0>(m_bounding_box_)[1], std::get<0>(m_bounding_box_)[2],
                   std::get<1>(m_bounding_box_)[0], std::get<1>(m_bounding_box_)[1], std::get<1>(m_bounding_box_)[2]);
}
std::shared_ptr<GeoObject> GeoObjectOCE::GetBoundary() const {
    DUMMY << "";
    return nullptr;
};

box_type GeoObjectOCE::GetBoundingBox() const { return m_bounding_box_; };

bool GeoObjectOCE::CheckIntersection(point_type const &x, Real tolerance) const {
    BRepBuilderAPI_MakeVertex vertex(make_point(x));
    BRepExtrema_DistShapeShape dist(*m_occ_shape_, vertex);
    dist.Perform();
    return dist.InnerSolution();
};
bool GeoObjectOCE::CheckIntersection(box_type const &b, Real tolerance) const {
    BRepPrimAPI_MakeBox box(make_point(std::get<0>(b)), make_point(std::get<1>(b)));
    BRepExtrema_DistShapeShape dist(*m_occ_shape_, box.Solid());
    dist.Perform();
    return dist.InnerSolution();
};
std::shared_ptr<GeoObject> GeoObjectOCE::GetUnion(std::shared_ptr<const GeoObject> const &g, Real tolerance) const {
    auto res = GeoObjectOCE::New();
    auto other = GeoObjectOCE::New(g);
    res->m_occ_shape_ = std::make_shared<TopoDS_Shape>(BRepAlgoAPI_Fuse(*m_occ_shape_, *other->m_occ_shape_));
    return res;
};
std::shared_ptr<GeoObject> GeoObjectOCE::GetDifference(std::shared_ptr<const GeoObject> const &g,
                                                       Real tolerance) const {
    auto res = GeoObjectOCE::New();
    auto other = GeoObjectOCE::New(g);
    res->m_occ_shape_ = std::make_shared<TopoDS_Shape>(BRepAlgoAPI_Cut(*m_occ_shape_, *other->m_occ_shape_));
    return res;
};
std::shared_ptr<GeoObject> GeoObjectOCE::GetIntersection(std::shared_ptr<const GeoObject> const &g,
                                                         Real tolerance) const {
    auto res = GeoObjectOCE::New();
    auto other = GeoObjectOCE::New(g);
    res->m_occ_shape_ = std::make_shared<TopoDS_Shape>(BRepAlgoAPI_Common(*m_occ_shape_, *other->m_occ_shape_));
    return res;
};

/********************************************************************************************************************/

struct IntersectionCurveSurfaceOCE : public IntersectionCurveSurface {
   public:
    std::string FancyTypeName() const override {
        return base_type::FancyTypeName() + "." + __STRING(IntersectionCurveSurface);
    }
    std::string GetRegisterName() const override { return RegisterName(); }
    static std::string RegisterName() { return "OCE"; }

   private:
    typedef IntersectionCurveSurface base_type;
    typedef IntersectionCurveSurfaceOCE this_type;
    static int _is_registered;

   protected:
    IntersectionCurveSurfaceOCE();

   public:
    ~IntersectionCurveSurfaceOCE() override;

    template <typename... Args>
    static std::shared_ptr<this_type> New(Args &&... args) {
        return std::shared_ptr<this_type>(new this_type(std::forward<Args>(args)...));
    };
    void Load() override;
    size_type Intersect(std::shared_ptr<const Edge> const &curve, std::vector<Real> *u) override;
    size_type Intersect(std::shared_ptr<const Edge> const &curve, std::vector<Real> *u) const override;

   private:
    BRepIntCurveSurface_Inter m_body_inter_;
};
int IntersectionCurveSurfaceOCE::_is_registered =
    Factory<IntersectionCurveSurface>::RegisterCreator<IntersectionCurveSurfaceOCE>();

IntersectionCurveSurfaceOCE::IntersectionCurveSurfaceOCE() = default;
IntersectionCurveSurfaceOCE::~IntersectionCurveSurfaceOCE() = default;

void IntersectionCurveSurfaceOCE::Load() {
    m_body_inter_.Load(*GeoObjectOCE::New(GetShape())->GetShape(), GetTolerance());
};

size_type IntersectionCurveSurfaceOCE::Intersect(std::shared_ptr<const Edge> const &curve, std::vector<Real> *u) const {
    UNIMPLEMENTED;
    return 0;
};

size_type IntersectionCurveSurfaceOCE::Intersect(std::shared_ptr<const Edge> const &curve, std::vector<Real> *u) {
    m_body_inter_.Init(make_geo_curve(curve));
    std::vector<Real> intersection_points;
    for (; m_body_inter_.More(); m_body_inter_.Next()) { intersection_points.push_back(m_body_inter_.W()); }
    std::sort(intersection_points.begin(), intersection_points.end());
    return intersection_points.size();
}
// void IntersectionCurveSurfaceTagNodeOCE(Array<Real> *vertex_tags, std::shared_ptr<const Chart> const &chart,
//                                        index_box_type const &m_idx_box, const std::shared_ptr<const GeoObject> &g,
//                                        int tag) {
//    Real tol = 0.01;
//    //    std::get<1>(m_idx_box) += 1;
//    box_type bnd_box = g->GetBoundingBox();
//    vector_type length = std::get<1>(bnd_box) - std::get<0>(bnd_box);
//    std::get<0>(bnd_box) -= 0.03 * length;
//    std::get<1>(bnd_box) += 0.03 * length;
//
//    gp_Pnt xlo{std::get<0>(bnd_box)[0], std::get<0>(bnd_box)[1], std::get<0>(bnd_box)[2]};
//    gp_Pnt xhi{std::get<1>(bnd_box)[0], std::get<1>(bnd_box)[1], std::get<1>(bnd_box)[2]};
//
//    BRepPrimAPI_MakeBox makeBox(xlo, xhi);
//    makeBox.Build();
//    auto box = makeBox.Shell();
//    BRepIntCurveSurface_Inter m_box_inter_;
//    m_box_inter_.Load(box, 0.0001);
//
//    BRepIntCurveSurface_Inter m_body_inter_;
//    m_body_inter_.Load(*oce_cast<TopoDS_Shape>(g), 0.0001);
//
//    //    Array<int, ZSFC<3>> vertex_tags(nullptr, m_idx_box);
//    //    vertex_tags.Clear();
//    //    std::map<EntityId, Real> m_edge_fraction;
//
//    for (int dir = 0; dir < 3; ++dir) {
//        index_tuple lo{0, 0, 0}, hi{0, 0, 0};
//        std::tie(lo, hi) = m_idx_box;
//        hi[dir] = lo[dir] + 1;
//        for (index_type i = lo[0]; i < hi[0]; ++i)
//            for (index_type j = lo[1]; j < hi[1]; ++j)
//                for (index_type k = lo[2]; k < hi[2]; ++k) {
//                    //                    point_type x_begin = chart->global_coordinates(i, j, k, 0b0);
//                    // start make_point is on the bounding box
//                    //                    {
//                    //                        index_tuple idx{i, j, k};
//                    //
//                    //                        index_type s0 = idx[make_dir];
//                    //                        Handle(Geom_Curve) c =
//                    //                            geometry::OCEShapeCast<Geom_Curve,
//                    //                            gCurve>::eval(*chart->GetAxis(x_begin, make_dir));
//                    //
//                    //                        m_box_inter_.Init(c);
//                    //
//                    //                        // if curve do not intersect with bounding box then continue to next
//                    curve
//                    //                        if (!m_box_inter_.More()) { continue; }
//                    //
//                    //                        bool is_first = true;
//                    //                        // search min intersection make_point
//                    //                        while (m_box_inter_.More()) {
//                    //                            index_tuple i1{0, 0, 0};
//                    //                            point_type x1{m_box_inter_.Pnt().X(), m_box_inter_.Pnt().Y(),
//                    //                            m_box_inter_.Pnt().Z()};
//                    //                            std::tie(i1, std::ignore) = chart->invert_global_coordinates(x1);
//                    //
//                    //                            if (is_first || i1[make_dir] < s0) {
//                    //                                s0 = i1[make_dir];
//                    //                                x_begin = x1;
//                    //                                is_first = false;
//                    //                            }
//                    //                            m_box_inter_.Next();
//                    //                        }
//                    //                    }
//
//                    point_type x_begin = chart->global_coordinates(0b0, i, j, k);
//                    Handle(Geom_Curve) c;  // = oce_cast<Geom_Curve>(chart->GetAxis(x_begin, x_begin));
//
//                    m_body_inter_.Init(c);
//
//                    std::vector<Real> intersection_points;
//                    for (; m_body_inter_.More(); m_body_inter_.Next()) {
//                        intersection_points.push_back(m_body_inter_.W());
//                    }
//
//                    std::sort(intersection_points.begin(), intersection_points.end());
//
//                    for (size_t n = 0; n < intersection_points.size(); n += 2) {
//                        gp_Pnt p0 = c->Value(intersection_points[n]);
//                        gp_Pnt p1 = c->Value(intersection_points[n + 1]);
//
//                        point_type x0{p0.X(), p0.Y(), p0.Z()};
//
//                        index_tuple i0{0, 0, 0};
//                        point_type r0{0, 0, 0};
//                        std::tie(i0, r0) = chart->invert_global_coordinates(x0);
//
//                        point_type x1{p1.X(), p1.Y(), p1.Z()};
//                        index_tuple i1{0, 0, 0};
//                        point_type r1{0, 0, 0};
//                        std::tie(i1, r1) = chart->invert_global_coordinates(x1);
//
//                        index_type s0 = std::max(i0[dir], std::get<0>(m_idx_box)[dir]);
//                        index_type s1 = std::min(i1[dir], std::get<1>(m_idx_box)[dir]);
//
//                        for (index_type s = i0[dir]; s <= i1[dir]; ++s) {
//                            index_tuple id{i, j, k};
//                            id[dir] = s;
//                            vertex_tags[0].Set(1, id);
//                        }
//                    }
//
//                    // std::cout << index_tuple{i, j, k} << "~" << idx << "~" << r <<
//                    // std::endl;
//                    // vertex_tags->SetEntity(count, idx);
//                    // std::cout << "DIR:" << make_dir << "\t" << m_idx_box << "\t" <<
//                    // index_tuple{i, j, k} << "\t" << idx;
//                    // if (!(CheckInSide(m_idx_box, idx))) {
//                    //    std::cout << std::endl;
//                    //    continue;
//                    // } else {
//                    //    std::cout << "\t" << (x) << "\t" << chart->inv_map(x) <<
//                    //    std::endl;
//                    // }
//                    //   edge_fraction[dir].SetEntity(r[make_dir], idx);
//                    //   vertex_tags->SetEntity(1, idx);
//                    //   idx[(make_dir + 1) % 3] -= 1;
//                    //   vertex_tags->SetEntity(1, idx);
//                    //   idx[(make_dir + 2) % 3] -= 1;
//                    //   vertex_tags->SetEntity(1, idx);
//                    //   idx[(make_dir + 1) % 3] += 1;
//                    //   vertex_tags->SetEntity(1, idx);
//                    //   index_tuple id{i, j, k};
//                    //   id[dir] = std::get<0>(l_coor)[make_dir];
//                    //   vertex_tags[0].SetEntity(make_dir + 1, id);
//                    //   id[(dir + 1) % 3] = idx[(make_dir + 1) % 3] - 1;
//                    //   vertex_tags[0].SetEntity(make_dir + 1, id);
//                    //   id[(make_dir + 2) % 3] = idx[(dir + 2) % 3] - 1;
//                    //   vertex_tags[0].SetEntity(make_dir + 1, id);
//                    //   id[(dir + 1) % 3] = idx[(make_dir + 1) % 3];
//                    //   vertex_tags[0].SetEntity(make_dir + 1, id);
//                    //   if (m_body_inter_.State() == TopAbs_IN) {
//                    //       s0 = std::max(std::get<0>(l_coor)[make_dir],
//                    //       std::get<0>(m_idx_box)[make_dir]);
//                    //   }
//                    //
//                    //   if (x[dir] < std::get<0>(m_box)[make_dir]) { continue; }
//                    //
//                    //   EntityId q;
//                    //   q.x = static_cast<int16_t>(std::get<0>(l_coor)[0]);
//                    //   q.y = static_cast<int16_t>(std::get<0>(l_coor)[1]);
//                    //   q.z = static_cast<int16_t>(std::get<0>(l_coor)[2]);
//                    //   q.w =
//                    //   static_cast<int16_t>(EntityIdCoder::m_sub_index_to_id_[EDGE][make_dir]);
//                    //   index_tuple idx{i, j, k};
//                    //   idx[make_dir] = std::get<0>(l_coor)[dir];
//                    //   edge_fraction[dir].SetEntity(std::get<1>(l_coor)[make_dir], idx);
//                    //
//
//                    //                        VERBOSE << "s0:" << s0 << " s1:" << s1 << std::endl;
//                    //                        if (x[dir] > std::get<1>(m_idx_box)[make_dir]) { break; }
//                }
//    }
//}
/********************************************************************************************************************/
int GeoEngineOCE::_is_registered = Factory<GeoEngineAPI>::RegisterCreator<GeoEngineOCE>(GeoEngineOCE::RegisterName());
struct GeoEngineOCE::pimpl_s {
    std::string m_prefix_ = "simpla";
    std::string m_ext_ = "stp";

    Handle(TDocStd_Document) aDoc;
    Handle(XCAFApp_Application) anApp;
    Handle(XCAFDoc_ShapeTool) myShapeTool;
};

GeoEngineOCE::GeoEngineOCE() : m_pimpl_(new pimpl_s){};
GeoEngineOCE::~GeoEngineOCE() {
    CloseFile();
    delete m_pimpl_;
};
void GeoEngineOCE::Deserialize(std::shared_ptr<simpla::data::DataEntry> const &cfg) {
    m_pimpl_->m_prefix_ = cfg->GetValue<std::string>("Path", m_pimpl_->m_prefix_);
};
std::shared_ptr<simpla::data::DataEntry> GeoEngineOCE::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue<std::string>("Prefix", m_pimpl_->m_prefix_);
    return res;
};

std::string GeoEngineOCE::GetFilePath() const { return m_pimpl_->m_prefix_ + "." + m_pimpl_->m_ext_; };
void GeoEngineOCE::OpenFile(std::string const &path) {
    CloseFile();
    base_type::OpenFile(path);

    auto pos = path.rfind('.');
    m_pimpl_->m_prefix_ = path.substr(0, pos);
    m_pimpl_->m_ext_ = path.substr(pos + 1);
    m_pimpl_->anApp = XCAFApp_Application::GetApplication();
    m_pimpl_->anApp->NewDocument("MDTV-XCAF", m_pimpl_->aDoc);
    m_pimpl_->myShapeTool = XCAFDoc_DocumentTool::ShapeTool(m_pimpl_->aDoc->Main());
};
void GeoEngineOCE::CloseFile() {
    if (IsOpened()) { FlushFile(); }
    m_pimpl_->m_prefix_ = "";
    base_type::CloseFile();
};
void GeoEngineOCE::FlushFile() {
    if (!IsOpened() || m_pimpl_->m_prefix_.empty()) {
    } else if (m_pimpl_->m_ext_ == "stp") {
        STEPCAFControl_Writer writer;
        if (!writer.Transfer(m_pimpl_->aDoc, STEPControl_AsIs)) {
            RUNTIME_ERROR << "The document cannot be translated or gives no result";
        }
        writer.Write((m_pimpl_->m_prefix_ + "." + m_pimpl_->m_ext_).c_str());
    } else if (m_pimpl_->m_ext_ == "igs") {
        IGESCAFControl_Writer writer;
        writer.Perform(m_pimpl_->aDoc, (m_pimpl_->m_prefix_ + "." + m_pimpl_->m_ext_).c_str());
    };
};
void GeoEngineOCE::Save(std::shared_ptr<const GeoObject> const &geo, std::string const &name_s) const {
    if (geo == nullptr) { return; }
    ASSERT(!m_pimpl_->m_prefix_.empty());
    CHECK(*geo->Serialize());
    auto oce_shape = GeoObjectOCE(geo).GetShape();
    std::string name = name_s;
    if (name.empty()) { name = std::to_string(oce_shape->HashCode(std::numeric_limits<int>::max())); }

    if (m_pimpl_->m_ext_ == "stl") {
        StlAPI_Writer().Write(*oce_shape, (m_pimpl_->m_prefix_ + "_" + name + ".stl").c_str(), true);
    } else {
        TDF_Label aLabel1 = m_pimpl_->myShapeTool->NewShape();
        Handle(TDataStd_Name) NameAttrib1 = new TDataStd_Name();
        NameAttrib1->Set(name.c_str());
        aLabel1.AddAttribute(NameAttrib1);
        m_pimpl_->myShapeTool->SetShape(aLabel1, *oce_shape);
        m_pimpl_->myShapeTool->UpdateAssembly(aLabel1);
    }
}
std::shared_ptr<GeoObject> GeoEngineOCE::Load(std::string const &name) const {
    ASSERT(!m_pimpl_->m_prefix_.empty());
    STEPControl_Reader reader;
    IFSelect_ReturnStatus stat = reader.ReadFile((m_pimpl_->m_prefix_ + "." + m_pimpl_->m_ext_).c_str());
    ASSERT(stat == IFSelect_RetDone);  // ExcMessage("Error in reading file!"));
    Standard_Boolean failsonly = Standard_False;
    IFSelect_PrintCount mode = IFSelect_ItemsByEntity;
    reader.PrintCheckLoad(failsonly, mode);
    Standard_Integer nRoots = reader.TransferRoots();
    ASSERT(nRoots > 0);
    VERBOSE << "STEP Object is loaded from " << m_pimpl_->m_prefix_ << "." << m_pimpl_->m_ext_ << "[" << nRoots << "]"
            << std::endl;
    return std::make_shared<GeoObjectOCE>(reader.OneShape());
}
bool GeoEngineOCE::CheckIntersection(std::shared_ptr<const GeoObject> const &g, point_type const &x,
                                     Real tolerance) const {
    bool res = false;
    if (g != nullptr) { res = GeoObjectOCE::New(g)->CheckIntersection(x, tolerance); }
    return res;
}
bool GeoEngineOCE::CheckIntersection(std::shared_ptr<const GeoObject> const &g, box_type const &b,
                                     Real tolerance) const {
    bool res = false;
    if (g != nullptr) { res = GeoObjectOCE::New(g)->CheckIntersection(b, tolerance); }
    return res;
}

std::shared_ptr<GeoObject> GeoEngineOCE::GetUnion(std::shared_ptr<const GeoObject> const &g0,
                                                  std::shared_ptr<const GeoObject> const &g1, Real tolerance) const {
    std::shared_ptr<GeoObject> res = nullptr;
    if (g0 != nullptr) { res = GeoObjectOCE::New(g0)->GetUnion(g1, tolerance); }
    return res;
}
std::shared_ptr<GeoObject> GeoEngineOCE::GetDifference(std::shared_ptr<const GeoObject> const &g0,
                                                       std::shared_ptr<const GeoObject> const &g1,
                                                       Real tolerance) const {
    std::shared_ptr<GeoObject> res = nullptr;
    if (g0 != nullptr) { res = GeoObjectOCE::New(g0)->GetDifference(g1, tolerance); }
    return res;
}
std::shared_ptr<GeoObject> GeoEngineOCE::GetIntersection(std::shared_ptr<const GeoObject> const &g0,
                                                         std::shared_ptr<const GeoObject> const &g1,
                                                         Real tolerance) const {
    std::shared_ptr<GeoObject> res = nullptr;
    if (g0 != nullptr) { res = GeoObjectOCE::New(g0)->GetIntersection(g1, tolerance); }
    return res;
}
}  // namespace geometry
}  // namespace simpla