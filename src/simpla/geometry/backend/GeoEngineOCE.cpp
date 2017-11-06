//
// Created by salmon on 17-11-1.
//

#include "GeoEngineOCE.h"
#include "../GeoObject.h"

#include <simpla/algebra/nTuple.ext.h>

#include <simpla/geometry/Chart.h>
#include <BRepAlgoAPI_Common.hxx>
#include <BRepAlgoAPI_Cut.hxx>
#include <BRepAlgoAPI_Fuse.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakePolygon.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepIntCurveSurface_Inter.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <BRepPrimAPI_MakeSphere.hxx>
#include <BRepPrimAPI_MakeTorus.hxx>
#include <Bnd_Box.hxx>
#include <GeomAdaptor_Curve.hxx>
#include <Geom_Circle.hxx>
#include <Geom_Curve.hxx>
#include <Geom_Ellipse.hxx>
#include <Geom_Hyperbola.hxx>
#include <Geom_Line.hxx>
#include <Geom_Parabola.hxx>
#include <Geom_Surface.hxx>
#include <Handle_TDocStd_Document.hxx>
#include <Handle_XCAFApp_Application.hxx>
#include <Handle_XCAFDoc_ShapeTool.hxx>
#include <Interface_Static.hxx>
#include <STEPCAFControl_Writer.hxx>
#include <STEPControl_Reader.hxx>
#include <Standard_Transient.hxx>
#include <StlAPI_Reader.hxx>
#include <TColStd_HSequenceOfTransient.hxx>
#include <TDataStd_Name.hxx>
#include <TDocStd_Document.hxx>
#include <TopoDS_Shape.hxx>
#include <XCAFApp_Application.hxx>
#include <XCAFDoc_DocumentTool.hxx>
#include <XCAFDoc_ShapeTool.hxx>
#include <gp_Quaternion.hxx>
#include "../Body.h"
#include "../Box.h"
#include "../Circle.h"
#include "../Curve.h"
#include "../Ellipse.h"
#include "../GeoObject.h"
#include "../Hyperbola.h"
#include "../IntersectionCurveSurface.h"
#include "../Line.h"
#include "../Parabola.h"
#include "../Polygon.h"
#include "../PrimitiveShape.h"
#include "../Sphere.h"
#include "../Surface.h"
#include "../Torus.h"
namespace simpla {
namespace geometry {
struct GeoObjectOCE;

namespace detail {
template <typename TDest, typename TSrc>
struct OCEShapeCast {
    static std::shared_ptr<TDest> eval(std::shared_ptr<const TSrc> const &s) {
        UNIMPLEMENTED;
        return nullptr;
    }
};

gp_Pnt make_point(point_type const &p0) { return gp_Pnt{p0[0], p0[1], p0[2]}; }
gp_Dir make_dir(vector_type const &p0) { return gp_Dir{p0[0], p0[1], p0[2]}; }
gp_Ax2 make_axe(point_type const &origin, vector_type const &z, vector_type const &x) {
    return gp_Ax2{make_point(origin), make_dir(z), make_dir(x)};
}
gp_Ax2 make_axis(Axis const &axis) { return gp_Ax2{make_point(axis.o), make_dir(axis.z), make_dir(axis.x)}; }
// template <>
// Handle(Geom_Curve) OCEGeometryCast<Geom_Curve, GeoObject>::eval(std::shared_ptr<const GeoObject> const &g) {
//    Handle(Geom_Curve) res;
//    if (auto c = std::dynamic_pointer_cast<Circle const>(g)) {
//        res = new Geom_Circle(gp_Ax2(make_point(c->GetAxis().o), make_dir(c->GetAxis().z), make_dir(c->GetAxis().x)),
//                              c->GetRadius());
//    } else if (auto l = std::dynamic_pointer_cast<Line const>(g)) {
//        res = new Geom_Line(make_point(l->GetAxis().o), make_dir(l->GetAxis().x));
//    } else {
//        UNIMPLEMENTED;
//    }
//    return res;
//};

template <>
std::shared_ptr<TopoDS_Shape> OCEShapeCast<TopoDS_Shape, Surface>::eval(std::shared_ptr<const Surface> const &g) {
    std::shared_ptr<TopoDS_Shape> res = nullptr;
    UNIMPLEMENTED;
    return res;
};
//
// template <>
// TopoDS_Shape *OCEShapeCast<TopoDS_Shape, GeoObject>::eval(GeoObject const &g);
// template <>
// Geom_Curve *OCEShapeCast<Geom_Curve, Curve>::eval(Curve const &c);
// template <>
// Geom_Surface *OCEShapeCast<Geom_Surface, Surface>::eval(Surface const &c);
// template <>
// TopoDS_Shape *OCEShapeCast<TopoDS_Shape, GeoObject>::eval(GeoObject const &g) {
//    auto *res = new TopoDS_Shape;
//    if (dynamic_cast<GeoObjectOCE const *>(&g) != nullptr) {
//        *res = dynamic_cast<GeoObjectOCE const &>(g).GetShape();
//    } else {
//        auto p = GeoObjectOCE::New(g);
//        *res = p->GetShape();
//    }
//    return res;
//}
// template <>
// Geom_Curve *OCEShapeCast<Geom_Curve, Curve>::eval(Curve const &c) {
//    Geom_Curve *res = nullptr;
//    if (dynamic_cast<Circle const *>(&c) != nullptr) {
//        auto const &l = dynamic_cast<Circle const &>(c);
//        //        res = new Geom_Circle(gp_Ax2(make_point(l.Origin()), dir(l.Normal()), make_dir(l.XAxis())),
//        //        l.Radius());
//    } else if (dynamic_cast<Line const *>(&c) != nullptr) {
//        auto const &l = dynamic_cast<Line const &>(c);
//        res = new Geom_Line(make_point(l.GetAxis().o), make_dir(l.GetAxis().x));
//    } else {
//        UNIMPLEMENTED;
//    }
//    return res;
//};
}

struct GeoObjectOCE : public GeoObject {
    SP_GEO_OBJECT_HEAD(GeoObjectOCE, GeoObject)

   public:
    explicit GeoObjectOCE(std::shared_ptr<TopoDS_Shape> const &shape);
    explicit GeoObjectOCE(GeoObject const &g);
    explicit GeoObjectOCE(std::shared_ptr<const GeoObject> const &g);

    int Load(std::string const &path, std::string const &name) override;
    int Save(std::string const &path, std::string const &name) const override;

    void Transform(Real scale, point_type const &location = point_type{0, 0, 0},
                   nTuple<Real, 4> const &rotate = nTuple<Real, 4>{0, 0, 0, 0});
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
bool GeoObjectOCE::_is_registered = simpla::Factory<GeoObject>::RegisterCreator<GeoObjectOCE>("oce") > 0;

namespace detail {

template <>
std::shared_ptr<TopoDS_Shape> OCEShapeCast<TopoDS_Shape, PrimitiveShape>::eval(
    std::shared_ptr<const PrimitiveShape> const &g) {
    std::shared_ptr<TopoDS_Shape> res = nullptr;
    if (auto box = std::dynamic_pointer_cast<const Box>(g)) {
        res = std::make_shared<TopoDS_Solid>(
            BRepPrimAPI_MakeBox(make_point(box->GetMinPoint()), make_point(box->GetMaxPoint())));
    } else if (auto sphere = std::dynamic_pointer_cast<const Sphere>(g)) {
        res = std::make_shared<TopoDS_Solid>(BRepPrimAPI_MakeSphere(make_axis(sphere->GetAxis()), sphere->GetRadius()));
    } else if (auto torus = std::dynamic_pointer_cast<const Torus>(g)) {
        res = std::make_shared<TopoDS_Solid>(
            BRepPrimAPI_MakeTorus(make_axis(torus->GetAxis()), torus->GetMajorRadius(), torus->GetMinorRadius()));
    } else {
        UNIMPLEMENTED;
    }
    return res;
};

template <>
std::shared_ptr<TopoDS_Shape> OCEShapeCast<TopoDS_Shape, Curve>::eval(std::shared_ptr<const Curve> const &g) {
    std::shared_ptr<TopoDS_Shape> res = nullptr;

    if (auto polygon = std::dynamic_pointer_cast<const Polygon>(g)) {
        BRepBuilderAPI_MakePolygon oce_polygon;
        for (auto const &p : polygon->data()) { oce_polygon.Add(gp_Pnt{p[0], p[1], 0}); }
        if (polygon->IsClosed()) { oce_polygon.Close(); }
        res = std::make_shared<TopoDS_Wire>(oce_polygon.Wire());
    } else {
        Handle(Geom_Curve) c;
        if (auto line = std::dynamic_pointer_cast<const Line>(g)) {
            c = new Geom_Line(make_point(line->GetStartPoint()), make_dir(line->GetEndPoint() - line->GetStartPoint()));
        } else if (auto circle = std::dynamic_pointer_cast<const Circle>(g)) {
            c = new Geom_Circle(make_axis(circle->GetAxis()), circle->GetRadius());
        } else if (auto ellipse = std::dynamic_pointer_cast<const Ellipse>(g)) {
            c = new Geom_Ellipse(make_axis(ellipse->GetAxis()), ellipse->GetMajorRadius(), ellipse->GetMinorRadius());
        } else if (auto hyperbola = std::dynamic_pointer_cast<const Hyperbola>(g)) {
            c = new Geom_Hyperbola(make_axis(hyperbola->GetAxis()), hyperbola->GetMajorRadius(),
                                   hyperbola->GetMinorRadius());
        } else if (auto parabola = std::dynamic_pointer_cast<const Parabola>(g)) {
            c = new Geom_Parabola(make_axis(parabola->GetAxis()), parabola->GetFocal());
        } else {
            UNIMPLEMENTED;
        }
        res = std::make_shared<TopoDS_Edge>(BRepBuilderAPI_MakeEdge(c));
    }
    return res;
};
template <>
std::shared_ptr<TopoDS_Shape> OCEShapeCast<TopoDS_Shape, GeoObject>::eval(std::shared_ptr<const GeoObject> const &g) {
    std::shared_ptr<TopoDS_Shape> res = nullptr;
    if (auto oce = std::dynamic_pointer_cast<GeoObjectOCE const>(g)) {
        res = oce->GetShape();
    } else if (auto c = std::dynamic_pointer_cast<Curve const>(g)) {
        res = OCEShapeCast<TopoDS_Shape, Curve>::eval(c);
    } else if (auto s = std::dynamic_pointer_cast<Surface const>(g)) {
        res = OCEShapeCast<TopoDS_Shape, Surface>::eval(s);
    } else if (auto b = std::dynamic_pointer_cast<Body const>(g)) {
        res = OCEShapeCast<TopoDS_Shape, Body>::eval(b);
    } else if (auto p = std::dynamic_pointer_cast<PrimitiveShape const>(g)) {
        res = OCEShapeCast<TopoDS_Shape, PrimitiveShape>::eval(p);
    } else {
        LOGGER << *g->Serialize();
        UNIMPLEMENTED;
    }
    return res;
};
}
template <typename TDest, typename TSrc>
auto oce_cast(std::shared_ptr<const TSrc> const &g) {
    return detail::OCEShapeCast<TDest, TSrc>::eval(g);
}

GeoObjectOCE::GeoObjectOCE() = default;
GeoObjectOCE::GeoObjectOCE(GeoObjectOCE const &shape) = default;
GeoObjectOCE::~GeoObjectOCE() = default;
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

    {}
    return SP_SUCCESS;
};

void GeoObjectOCE::Transform(Real scale, point_type const &location, nTuple<Real, 4> const &rotate) {
    m_occ_shape_ = TransformShape(m_occ_shape_, scale, location, rotate);
}
std::shared_ptr<data::DataNode> GeoObjectOCE::Serialize() const {
    auto res = base_type::Serialize();
    res->SetValue("HashCode", m_occ_shape_->HashCode(std::numeric_limits<int>::max()));
    return res;
};
void GeoObjectOCE::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    auto tdb = std::dynamic_pointer_cast<const data::DataNode>(cfg);
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
int GeoObjectOCE::Load(std::string const &path, std::string const &name) {
    m_occ_shape_ = LoadOCEShape(path, name);
    return m_occ_shape_ == nullptr ? SP_FAILED : SP_SUCCESS;
};

int GeoObjectOCE::Save(std::string const &path, std::string const &name) const {
    return SaveOCEShape(m_occ_shape_, path, name);
}

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
    BRepBuilderAPI_MakeVertex vertex(detail::make_point(x));
    BRepExtrema_DistShapeShape dist(*m_occ_shape_, vertex);
    dist.Perform();
    return dist.InnerSolution();
};
bool GeoObjectOCE::CheckIntersection(box_type const &b, Real tolerance) const {
    BRepPrimAPI_MakeBox box(detail::make_point(std::get<0>(b)), detail::make_point(std::get<1>(b)));
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
    static std::string RegisterName_s() { return __STRING(OCE); }
    std::string RegisterName() const override { return RegisterName_s(); }

   private:
    typedef IntersectionCurveSurface base_type;
    typedef IntersectionCurveSurfaceOCE this_type;
    static int _is_registered;

   protected:
    IntersectionCurveSurfaceOCE();

   public:
    ~IntersectionCurveSurfaceOCE() override;
    void SetUp(std::shared_ptr<const Surface> const &, Real tolerance) override;
    size_type Intersect(std::shared_ptr<const Curve> const &curve, std::vector<Real> *u) const override;

   private:
    BRepIntCurveSurface_Inter m_body_inter_;
};
int IntersectionCurveSurfaceOCE::_is_registered =
    Factory<IntersectionCurveSurface>::RegisterCreator<IntersectionCurveSurfaceOCE>(
        IntersectionCurveSurfaceOCE::RegisterName_s());

IntersectionCurveSurfaceOCE::IntersectionCurveSurfaceOCE() = default;
IntersectionCurveSurfaceOCE::~IntersectionCurveSurfaceOCE() = default;

void IntersectionCurveSurfaceOCE::SetUp(std::shared_ptr<const Surface> const &s, Real tolerance) {}
size_type IntersectionCurveSurfaceOCE::Intersect(std::shared_ptr<const Curve> const &curve,
                                                 std::vector<Real> *u) const {
    size_type count = 0;
    //    Handle(Geom_Curve) c = geometry::detail::OCEShapeCast<Geom_Curve, GeoObject>::eval(*curve);
    //
    //    m_body_inter_.Init(c);
    //
    //    std::vector<Real> intersection_points;
    //    for (; m_body_inter_.More(); m_body_inter_.Next()) {
    //        intersection_points.push_back(m_body_inter_.W());
    //        ++count;
    //    }
    //
    //    std::sort(intersection_points.begin(), intersection_points.end());
    return count;
}
void IntersectionCurveSurfaceTagNodeOCE(Array<Real> *vertex_tags, std::shared_ptr<const Chart> const &chart,
                                        index_box_type const &m_idx_box, const std::shared_ptr<const GeoObject> &g,
                                        int tag) {
    auto const &scale = chart->GetScale();
    Real tol = std::sqrt(dot(scale, scale) * 0.01);
    //    std::get<1>(m_idx_box) += 1;
    box_type bnd_box = g->GetBoundingBox();
    vector_type length = std::get<1>(bnd_box) - std::get<0>(bnd_box);
    std::get<0>(bnd_box) -= 0.03 * length;
    std::get<1>(bnd_box) += 0.03 * length;

    gp_Pnt xlo{std::get<0>(bnd_box)[0], std::get<0>(bnd_box)[1], std::get<0>(bnd_box)[2]};
    gp_Pnt xhi{std::get<1>(bnd_box)[0], std::get<1>(bnd_box)[1], std::get<1>(bnd_box)[2]};

    BRepPrimAPI_MakeBox makeBox(xlo, xhi);
    makeBox.Build();
    auto box = makeBox.Shell();
    BRepIntCurveSurface_Inter m_box_inter_;
    m_box_inter_.Load(box, 0.0001);

    BRepIntCurveSurface_Inter m_body_inter_;
    m_body_inter_.Load(*oce_cast<TopoDS_Shape>(g), 0.0001);

    //    Array<int, ZSFC<3>> vertex_tags(nullptr, m_idx_box);
    //    vertex_tags.Clear();
    //    std::map<EntityId, Real> m_edge_fraction;

    for (int dir = 0; dir < 3; ++dir) {
        index_tuple lo{0, 0, 0}, hi{0, 0, 0};
        std::tie(lo, hi) = m_idx_box;
        hi[dir] = lo[dir] + 1;
        for (index_type i = lo[0]; i < hi[0]; ++i)
            for (index_type j = lo[1]; j < hi[1]; ++j)
                for (index_type k = lo[2]; k < hi[2]; ++k) {
                    //                    point_type x_begin = chart->global_coordinates(i, j, k, 0b0);
                    // start make_point is on the bounding box
                    //                    {
                    //                        index_tuple idx{i, j, k};
                    //
                    //                        index_type s0 = idx[make_dir];
                    //                        Handle(Geom_Curve) c =
                    //                            geometry::detail::OCEShapeCast<Geom_Curve,
                    //                            Curve>::eval(*chart->GetAxis(x_begin, make_dir));
                    //
                    //                        m_box_inter_.Init(c);
                    //
                    //                        // if curve do not intersect with bounding box then continue to next curve
                    //                        if (!m_box_inter_.More()) { continue; }
                    //
                    //                        bool is_first = true;
                    //                        // search min intersection make_point
                    //                        while (m_box_inter_.More()) {
                    //                            index_tuple i1{0, 0, 0};
                    //                            point_type x1{m_box_inter_.Pnt().X(), m_box_inter_.Pnt().Y(),
                    //                            m_box_inter_.Pnt().Z()};
                    //                            std::tie(i1, std::ignore) = chart->invert_global_coordinates(x1);
                    //
                    //                            if (is_first || i1[make_dir] < s0) {
                    //                                s0 = i1[make_dir];
                    //                                x_begin = x1;
                    //                                is_first = false;
                    //                            }
                    //                            m_box_inter_.Next();
                    //                        }
                    //                    }

                    point_type x_begin = chart->global_coordinates(0b0, i, j, k);
                    Handle(Geom_Curve) c;  // = oce_cast<Geom_Curve>(chart->GetAxis(x_begin, x_begin));

                    m_body_inter_.Init(c);

                    std::vector<Real> intersection_points;
                    for (; m_body_inter_.More(); m_body_inter_.Next()) {
                        intersection_points.push_back(m_body_inter_.W());
                    }

                    std::sort(intersection_points.begin(), intersection_points.end());

                    for (size_t n = 0; n < intersection_points.size(); n += 2) {
                        gp_Pnt p0 = c->Value(intersection_points[n]);
                        gp_Pnt p1 = c->Value(intersection_points[n + 1]);

                        point_type x0{p0.X(), p0.Y(), p0.Z()};

                        index_tuple i0{0, 0, 0};
                        point_type r0{0, 0, 0};
                        std::tie(i0, r0) = chart->invert_global_coordinates(x0);

                        point_type x1{p1.X(), p1.Y(), p1.Z()};
                        index_tuple i1{0, 0, 0};
                        point_type r1{0, 0, 0};
                        std::tie(i1, r1) = chart->invert_global_coordinates(x1);

                        index_type s0 = std::max(i0[dir], std::get<0>(m_idx_box)[dir]);
                        index_type s1 = std::min(i1[dir], std::get<1>(m_idx_box)[dir]);

                        for (index_type s = i0[dir]; s <= i1[dir]; ++s) {
                            index_tuple id{i, j, k};
                            id[dir] = s;
                            vertex_tags[0].Set(1, id);
                        }
                    }

                    // std::cout << index_tuple{i, j, k} << "~" << idx << "~" << r <<
                    // std::endl;
                    // vertex_tags->SetEntity(count, idx);
                    // std::cout << "DIR:" << make_dir << "\t" << m_idx_box << "\t" <<
                    // index_tuple{i, j, k} << "\t" << idx;
                    // if (!(CheckInSide(m_idx_box, idx))) {
                    //    std::cout << std::endl;
                    //    continue;
                    // } else {
                    //    std::cout << "\t" << (x) << "\t" << chart->inv_map(x) <<
                    //    std::endl;
                    // }
                    //   edge_fraction[dir].SetEntity(r[make_dir], idx);
                    //   vertex_tags->SetEntity(1, idx);
                    //   idx[(make_dir + 1) % 3] -= 1;
                    //   vertex_tags->SetEntity(1, idx);
                    //   idx[(make_dir + 2) % 3] -= 1;
                    //   vertex_tags->SetEntity(1, idx);
                    //   idx[(make_dir + 1) % 3] += 1;
                    //   vertex_tags->SetEntity(1, idx);
                    //   index_tuple id{i, j, k};
                    //   id[dir] = std::get<0>(l_coor)[make_dir];
                    //   vertex_tags[0].SetEntity(make_dir + 1, id);
                    //   id[(dir + 1) % 3] = idx[(make_dir + 1) % 3] - 1;
                    //   vertex_tags[0].SetEntity(make_dir + 1, id);
                    //   id[(make_dir + 2) % 3] = idx[(dir + 2) % 3] - 1;
                    //   vertex_tags[0].SetEntity(make_dir + 1, id);
                    //   id[(dir + 1) % 3] = idx[(make_dir + 1) % 3];
                    //   vertex_tags[0].SetEntity(make_dir + 1, id);
                    //   if (m_body_inter_.State() == TopAbs_IN) {
                    //       s0 = std::max(std::get<0>(l_coor)[make_dir],
                    //       std::get<0>(m_idx_box)[make_dir]);
                    //   }
                    //
                    //   if (x[dir] < std::get<0>(m_box)[make_dir]) { continue; }
                    //
                    //   EntityId q;
                    //   q.x = static_cast<int16_t>(std::get<0>(l_coor)[0]);
                    //   q.y = static_cast<int16_t>(std::get<0>(l_coor)[1]);
                    //   q.z = static_cast<int16_t>(std::get<0>(l_coor)[2]);
                    //   q.w =
                    //   static_cast<int16_t>(EntityIdCoder::m_sub_index_to_id_[EDGE][make_dir]);
                    //   index_tuple idx{i, j, k};
                    //   idx[make_dir] = std::get<0>(l_coor)[dir];
                    //   edge_fraction[dir].SetEntity(std::get<1>(l_coor)[make_dir], idx);
                    //

                    //                        VERBOSE << "s0:" << s0 << " s1:" << s1 << std::endl;
                    //                        if (x[dir] > std::get<1>(m_idx_box)[make_dir]) { break; }
                }
    }
}
/********************************************************************************************************************/
bool GeoEngineOCE::_is_registered = Factory<GeoEngine>::RegisterCreator<GeoEngineOCE>(GeoEngineOCE::RegisterName());

GeoEngineOCE::GeoEngineOCE() = default;
GeoEngineOCE::~GeoEngineOCE() = default;

void GeoEngineOCE::SaveAPI(std::shared_ptr<const GeoObject> const &geo, std::string const &path,
                           std::string const &name) const {
    GeoObjectOCE(geo).Save(path, name);
}
std::shared_ptr<const GeoObject> GeoEngineOCE::LoadAPI(std::string const &path, std::string const &name) const {
    auto res = std::make_shared<GeoObjectOCE>();
    res->Load(path, name);
    return res;
}
// std::shared_ptr<GeoObject> GeoEngineOCE::GetBoundaryAPI(std::shared_ptr<const GeoObject> const &) const {}
bool GeoEngineOCE::CheckIntersectionAPI(std::shared_ptr<const GeoObject> const &g, point_type const &x,
                                        Real tolerance) const {
    bool res = false;
    if (g != nullptr) { res = GeoObjectOCE::New(g)->CheckIntersection(x, tolerance); }
    return res;
}
bool GeoEngineOCE::CheckIntersectionAPI(std::shared_ptr<const GeoObject> const &g, box_type const &b,
                                        Real tolerance) const {
    bool res = false;
    if (g != nullptr) { res = GeoObjectOCE::New(g)->CheckIntersection(b, tolerance); }
    return res;
}

std::shared_ptr<GeoObject> GeoEngineOCE::GetUnionAPI(std::shared_ptr<const GeoObject> const &g0,
                                                     std::shared_ptr<const GeoObject> const &g1, Real tolerance) const {
    std::shared_ptr<GeoObject> res = nullptr;
    if (g0 != nullptr) { res = GeoObjectOCE::New(g0)->GetUnion(g1, tolerance); }
    return res;
}
std::shared_ptr<GeoObject> GeoEngineOCE::GetDifferenceAPI(std::shared_ptr<const GeoObject> const &g0,
                                                          std::shared_ptr<const GeoObject> const &g1,
                                                          Real tolerance) const {
    std::shared_ptr<GeoObject> res = nullptr;
    if (g0 != nullptr) { res = GeoObjectOCE::New(g0)->GetDifference(g1, tolerance); }
    return res;
}
std::shared_ptr<GeoObject> GeoEngineOCE::GetIntersectionAPI(std::shared_ptr<const GeoObject> const &g0,
                                                            std::shared_ptr<const GeoObject> const &g1,
                                                            Real tolerance) const {
    std::shared_ptr<GeoObject> res = nullptr;
    if (g0 != nullptr) { res = GeoObjectOCE::New(g0)->GetIntersection(g1, tolerance); }
    return res;
}
}  // namespace geometry
}  // namespace simpla