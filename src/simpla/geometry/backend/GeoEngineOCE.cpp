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
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepIntCurveSurface_Inter.hxx>
#include <BRepPrimAPI_MakeBox.hxx>
#include <Bnd_Box.hxx>
#include <GeomAdaptor_Curve.hxx>
#include <Geom_Circle.hxx>
#include <Geom_Curve.hxx>
#include <Geom_Line.hxx>
#include <Geom_Surface.hxx>
#include <Interface_Static.hxx>
#include <STEPControl_Reader.hxx>
#include <Standard_Transient.hxx>
#include <StlAPI_Reader.hxx>
#include <TColStd_HSequenceOfTransient.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Quaternion.hxx>

#include "../Circle.h"
#include "../GeoObject.h"
#include "../IntersectionCurveSurface.h"
#include "../Line.h"
namespace simpla {
namespace geometry {

class Surface;
class Curve;
namespace detail {
template <typename TDest, typename TSrc, typename Enable = void>
struct OCCCast {
    static TDest *eval(TSrc const &s) { return nullptr; }
};
//
//template <>
//TopoDS_Shape *OCCCast<TopoDS_Shape, GeoObject>::eval(GeoObject const &g);
//template <>
//Geom_Curve *OCCCast<Geom_Curve, Curve>::eval(Curve const &c);
//template <>
//Geom_Surface *OCCCast<Geom_Surface, Surface>::eval(Surface const &c);

gp_Pnt make_point(point_type const &p0) { return gp_Pnt{p0[0], p0[1], p0[2]}; }
gp_Dir make_dir(vector_type const &p0) { return gp_Dir{p0[0], p0[1], p0[2]}; }
// template <>
// TopoDS_Shape *OCCCast<TopoDS_Shape, GeoObject>::eval(GeoObject const &g) {
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
// Geom_Curve *OCCCast<Geom_Curve, Curve>::eval(Curve const &c) {
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

template <typename TDest, typename TSrc>
TDest *occ_cast(TSrc const &g) {
    return detail::OCCCast<TDest, TSrc>::eval(g);
}

struct GeoObjectOCE : public GeoObject {
    SP_GEO_OBJECT_HEAD(GeoObjectOCE, GeoObject)

   public:
    GeoObjectOCE();
    GeoObjectOCE(GeoObjectOCE const &shape);
    ~GeoObjectOCE() override;

    GeoObjectOCE(TopoDS_Shape shape);
    GeoObjectOCE(GeoObject const &g);

    std::string ClassName() const override { return "GeoObjectOCE"; }
    static std::string RegisterName_s() { return "GeoObjectOCE"; }

    int Load(std::string const &authority, std::string const &path, std::string const &query,
             std::string const &fragment) override;

    void Transform(Real scale, point_type const &location = point_type{0, 0, 0},
                   nTuple<Real, 4> const &rotate = nTuple<Real, 4>{0, 0, 0, 0});
    void DoUpdate();

    TopoDS_Shape const &GetShape() const;
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
    TopoDS_Shape m_occ_shape_;
    box_type m_bounding_box_{{0, 0, 0}, {0, 0, 0}};

    Bnd_Box m_occ_box_;
};

bool GeoObjectOCE::_is_registered = simpla::Factory<GeoObject>::RegisterCreator<GeoObjectOCE>("oce") > 0;

GeoObjectOCE::GeoObjectOCE() = default;
GeoObjectOCE::GeoObjectOCE(GeoObjectOCE const &shape) = default;
GeoObjectOCE::~GeoObjectOCE() = default;
GeoObjectOCE::GeoObjectOCE(TopoDS_Shape shape) : m_occ_shape_(std::move(shape)) {}

GeoObjectOCE::GeoObjectOCE(GeoObject const &g) : GeoObjectOCE() {
    if (auto p = dynamic_cast<GeoObjectOCE const *>(&g)) {
        m_occ_shape_ = p->m_occ_shape_;
    } else {
    }
    DoUpdate();
};

TopoDS_Shape const &GeoObjectOCE::GetShape() const { return m_occ_shape_; }
Bnd_Box const &GeoObjectOCE::GetOCCBoundingBox() const { return m_occ_box_; }

TopoDS_Shape ReadSTEP(std::string const &file_name) {
    STEPControl_Reader reader;

    IFSelect_ReturnStatus stat = reader.ReadFile(file_name.c_str());

    ASSERT(stat == IFSelect_RetDone);  // ExcMessage("Error in reading file!"));

    Standard_Boolean failsonly = Standard_False;
    IFSelect_PrintCount mode = IFSelect_ItemsByEntity;
    reader.PrintCheckLoad(failsonly, mode);

    Standard_Integer nRoots = reader.TransferRoots();

    ASSERT(nRoots > 0);  //, 262 ExcMessage("Read nothing from file."));
    VERBOSE << "STEP Object is loaded from " << file_name << "[" << nRoots << "]" << std::endl;
    return reader.OneShape();
}
TopoDS_Shape ReadSTL(std::string const &file_name) {
    StlAPI_Reader reader;
    TopoDS_Shape shape;
    reader.Read(shape, file_name.c_str());
    return shape;
}

TopoDS_Shape TransformShape(TopoDS_Shape const &shape, Real scale, point_type const &location,
                            nTuple<Real, 4> const &rotate) {
    // Handle STEP Scale here.
    gp_Pnt origin{location[0], location[1], location[2]};
    gp_Quaternion rot_v{rotate[0], rotate[1], rotate[2], rotate[3]};
    gp_Trsf transf;
    transf.SetScale(origin, scale);
    //    transf.SetRotation(rot_v);
    BRepBuilderAPI_Transform trans(shape, transf);

    return trans.Shape();
}

TopoDS_Shape LoadShape(std::string const &file_name) {
    TopoDS_Shape res;
    std::string ext = file_name.substr(file_name.rfind('.') + 1);
    if (ext == "step" || ext == "stp") {
        res = ReadSTEP(file_name);
    } else if (ext == "stl") {
        res = ReadSTL(file_name);
    }
    return res;
};

void GeoObjectOCE::Transform(Real scale, point_type const &location, nTuple<Real, 4> const &rotate) {
    m_occ_shape_ = TransformShape(m_occ_shape_, scale, location, rotate);
}
std::shared_ptr<data::DataNode> GeoObjectOCE::Serialize() const {
    UNIMPLEMENTED;
    return base_type::Serialize();
};
void GeoObjectOCE::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    auto tdb = std::dynamic_pointer_cast<const data::DataNode>(cfg);
    if (tdb != nullptr) {
        m_occ_shape_ = TransformShape(
            LoadShape(tdb->GetValue<std::string>("File", "")),
            tdb->GetValue<Real>("Scale", 1.0e-3),            // default length unit is "m", STEP length unit is "mm"
            tdb->GetValue("Location", point_type{0, 0, 0}),  //
            tdb->GetValue("Rotation", nTuple<Real, 4>{0, 0, 0, 0}));
    }
    DoUpdate();
    VERBOSE << " [ Bounding Box :" << m_bounding_box_ << "]" << std::endl;
};
int GeoObjectOCE::Load(std::string const &authority, std::string const &path, std::string const &query,
                       std::string const &fragment) {
    m_occ_shape_ = LoadShape(path);
    return m_occ_shape_.IsNull() ? SP_FAILED : SP_SUCCESS;
};
void GeoObjectOCE::DoUpdate() {
    ASSERT(!m_occ_shape_.IsNull());
    BRepBndLib::Add(m_occ_shape_, m_occ_box_);
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
    BRepExtrema_DistShapeShape dist(vertex, m_occ_shape_);
    dist.Perform();
    return dist.InnerSolution();
};
bool GeoObjectOCE::CheckIntersection(box_type const &b, Real tolerance) const {
    BRepPrimAPI_MakeBox box(detail::make_point(std::get<0>(b)), detail::make_point(std::get<1>(b)));
    BRepExtrema_DistShapeShape dist(m_occ_shape_, box.Solid());
    dist.Perform();
    return dist.InnerSolution();
};
std::shared_ptr<GeoObject> GeoObjectOCE::GetUnion(std::shared_ptr<const GeoObject> const &g, Real tolerance) const {
    auto res = GeoObjectOCE::New();
    auto other = GeoObjectOCE::New(*g);
    res->m_occ_shape_ = BRepAlgoAPI_Fuse(m_occ_shape_, other->m_occ_shape_);
    return res;
};
std::shared_ptr<GeoObject> GeoObjectOCE::GetDifference(std::shared_ptr<const GeoObject> const &g,
                                                       Real tolerance) const {
    auto res = GeoObjectOCE::New();
    auto other = GeoObjectOCE::New(*g);
    res->m_occ_shape_ = BRepAlgoAPI_Cut(m_occ_shape_, other->m_occ_shape_);
    return res;
};
std::shared_ptr<GeoObject> GeoObjectOCE::GetIntersection(std::shared_ptr<const GeoObject> const &g,
                                                         Real tolerance) const {
    auto res = GeoObjectOCE::New();
    auto other = GeoObjectOCE::New(*g);
    res->m_occ_shape_ = BRepAlgoAPI_Common(m_occ_shape_, other->m_occ_shape_);
    return res;
};

/********************************************************************************************************************/

struct IntersectionCurveSurfaceOCE : public IntersectionCurveSurface {
    SP_GEO_ENGINE_HEAD(OCE, IntersectionCurveSurface)

   public:
    void SetUp(std::shared_ptr<const Surface> const &, Real tolerance) override;
    size_type Intersect(std::shared_ptr<const Curve> const &curve, std::vector<Real> *u) const override;

   private:
    BRepIntCurveSurface_Inter m_body_inter_;
};
REGISTER_CREATOR1(IntersectionCurveSurfaceOCE);

IntersectionCurveSurfaceOCE::IntersectionCurveSurfaceOCE() = default;
IntersectionCurveSurfaceOCE::~IntersectionCurveSurfaceOCE() = default;

void IntersectionCurveSurfaceOCE::SetUp(std::shared_ptr<const Surface> const &s, Real tolerance) {}
size_type IntersectionCurveSurfaceOCE::Intersect(std::shared_ptr<const Curve> const &curve,
                                                 std::vector<Real> *u) const {
    size_type count = 0;
    //    Handle(Geom_Curve) c = geometry::detail::OCCCast<Geom_Curve, GeoObject>::eval(*curve);
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
    m_body_inter_.Load(*geometry::occ_cast<TopoDS_Shape>(*g), 0.0001);

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
                    //                            geometry::detail::OCCCast<Geom_Curve,
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
                    Handle(Geom_Curve) c =
                        geometry::detail::OCCCast<Geom_Curve, GeoObject>::eval(*chart->GetAxis(x_begin, x_begin));

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
REGISTER_CREATOR1(GeoEngineOCE);
GeoEngineOCE::GeoEngineOCE() = default;
GeoEngineOCE::~GeoEngineOCE() = default;
// std::shared_ptr<GeoObject> GeoEngineOCE::GetBoundaryInterface(std::shared_ptr<const GeoObject> const &) const {}
bool GeoEngineOCE::CheckIntersectionInterface(std::shared_ptr<const GeoObject> const &g, point_type const &x,
                                              Real tolerance) const {
    bool res = false;
    if (g != nullptr) { res = GeoObjectOCE::New(*g)->CheckIntersection(x, tolerance); }
    return res;
}
bool GeoEngineOCE::CheckIntersectionInterface(std::shared_ptr<const GeoObject> const &g, box_type const &b,
                                              Real tolerance) const {
    bool res = false;
    if (g != nullptr) { res = GeoObjectOCE::New(*g)->CheckIntersection(b, tolerance); }
    return res;
}

std::shared_ptr<GeoObject> GeoEngineOCE::GetUnionInterface(std::shared_ptr<const GeoObject> const &g0,
                                                           std::shared_ptr<const GeoObject> const &g1,
                                                           Real tolerance) const {
    std::shared_ptr<GeoObject> res = nullptr;
    if (g0 != nullptr) { res = GeoObjectOCE::New(*g0)->GetUnion(g1, tolerance); }
    return res;
}
std::shared_ptr<GeoObject> GeoEngineOCE::GetDifferenceInterface(std::shared_ptr<const GeoObject> const &g0,
                                                                std::shared_ptr<const GeoObject> const &g1,
                                                                Real tolerance) const {
    std::shared_ptr<GeoObject> res = nullptr;
    if (g0 != nullptr) { res = GeoObjectOCE::New(*g0)->GetDifference(g1, tolerance); }
    return res;
}
std::shared_ptr<GeoObject> GeoEngineOCE::GetIntersectionInterface(std::shared_ptr<const GeoObject> const &g0,
                                                                  std::shared_ptr<const GeoObject> const &g1,
                                                                  Real tolerance) const {
    std::shared_ptr<GeoObject> res = nullptr;
    if (g0 != nullptr) { res = GeoObjectOCE::New(*g0)->GetIntersection(g1, tolerance); }
    return res;
}
}  // namespace geometry
}  // namespace simpla