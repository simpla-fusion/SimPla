//
// Created by salmon on 17-7-27.
//

#include "GeoObjectOCE.h"

#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepBuilderAPI_Transform.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <Bnd_Box.hxx>
#include <Interface_Static.hxx>
#include <STEPControl_Reader.hxx>
#include <StlAPI_Reader.hxx>
#include <TColStd_HSequenceOfTransient.hxx>
#include <TopoDS_Shape.hxx>
#include <gp_Quaternion.hxx>

#include "simpla/utilities/SPDefines.h"
namespace simpla {
namespace geometry {

SP_OBJECT_REGISTER(GeoObjectOCE)

struct GeoObjectOCE::pimpl_s {
    Real m_measure_ = SP_SNaN;
    TopoDS_Shape m_occ_shape_;
    box_type m_bounding_box_{{0, 0, 0}, {0, 0, 0}};

    Bnd_Box m_occ_box_;
};
GeoObjectOCE::GeoObjectOCE() : m_pimpl_(new pimpl_s){};

GeoObjectOCE::GeoObjectOCE(GeoObject const &g) : GeoObjectOCE() {
    if (dynamic_cast<GeoObjectOCE const *>(&g) == nullptr) {
        UNIMPLEMENTED;
    } else {
        m_pimpl_->m_occ_shape_ = dynamic_cast<GeoObjectOCE const &>(g).m_pimpl_->m_occ_shape_;
        DoUpdate();
    }
};

GeoObjectOCE::GeoObjectOCE(TopoDS_Shape const &shape) : GeoObjectOCE() {
    m_pimpl_->m_occ_shape_ = shape;
    DoUpdate();
}

GeoObjectOCE::~GeoObjectOCE() { delete m_pimpl_; };

TopoDS_Shape const &GeoObjectOCE::GetShape() const { return m_pimpl_->m_occ_shape_; }
Bnd_Box const &GeoObjectOCE::GetOCCBoundingBox() const { return m_pimpl_->m_occ_box_; }

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
    TopoDS_Shape tmp = m_pimpl_->m_occ_shape_;
    m_pimpl_->m_occ_shape_ = TransformShape(tmp, scale, location, rotate);
}
std::shared_ptr<data::DataNode> GeoObjectOCE::Serialize() const { return base_type::Serialize(); };
void GeoObjectOCE::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {
    base_type::Deserialize(cfg);
    auto tdb = std::dynamic_pointer_cast<const data::DataNode>(cfg);
    if (tdb != nullptr) {
        m_pimpl_->m_occ_shape_ = TransformShape(
            LoadShape(tdb->GetValue<std::string>("File", "")),
            tdb->GetValue<Real>("Scale", 1.0e-3),            // default length unit is "m", STEP length unit is "mm"
            tdb->GetValue("Location", point_type{0, 0, 0}),  //
            tdb->GetValue("Rotation", nTuple<Real, 4>{0, 0, 0, 0}));
    }
    DoUpdate();
    VERBOSE << " [ Bounding Box :" << m_pimpl_->m_bounding_box_ << "]" << std::endl;
};

void GeoObjectOCE::Load(std::string const &file_name) { m_pimpl_->m_occ_shape_ = LoadShape(file_name); };
void GeoObjectOCE::DoUpdate() {
    BRepBndLib::Add(m_pimpl_->m_occ_shape_, m_pimpl_->m_occ_box_);
    m_pimpl_->m_occ_box_.Get(std::get<0>(m_pimpl_->m_bounding_box_)[0], std::get<0>(m_pimpl_->m_bounding_box_)[1],
                             std::get<0>(m_pimpl_->m_bounding_box_)[2], std::get<1>(m_pimpl_->m_bounding_box_)[0],
                             std::get<1>(m_pimpl_->m_bounding_box_)[1], std::get<1>(m_pimpl_->m_bounding_box_)[2]);
}

box_type GeoObjectOCE::GetBoundingBox() const { return m_pimpl_->m_bounding_box_; };

bool GeoObjectOCE::CheckInside(point_type const &x, Real tolerance) const {
    //    VERBOSE << m_pimpl_->m_bounding_box_ << (x) << std::endl;
    gp_Pnt p(x[0], x[1], x[2]);
    //    gp_Pnt p(0,0,0);
    BRepBuilderAPI_MakeVertex vertex(p);
    BRepExtrema_DistShapeShape dist(vertex, m_pimpl_->m_occ_shape_);
    dist.Perform();
    return dist.InnerSolution();
};
}
}  // namespace simpla