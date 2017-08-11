//
// Created by salmon on 17-7-27.
//

#include "GeoObjectOCC.h"

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

REGISTER_CREATOR(GeoObjectOCC, occ)

struct GeoObjectOCC::pimpl_s {
    Real m_measure_ = SNaN;
    TopoDS_Shape m_occ_shape_;
    box_type m_bounding_box_{{0, 0, 0}, {0, 0, 0}};

    Bnd_Box m_occ_box_;
};
GeoObjectOCC::GeoObjectOCC() : m_pimpl_(new pimpl_s){};

GeoObjectOCC::GeoObjectOCC(GeoObject const &g) : GeoObjectOCC() {
    if (dynamic_cast<GeoObjectOCC const *>(&g) == nullptr) {
        UNIMPLEMENTED;
    } else {
        m_pimpl_->m_occ_shape_ = dynamic_cast<GeoObjectOCC const &>(g).m_pimpl_->m_occ_shape_;
        Update();
    }
};
GeoObjectOCC::GeoObjectOCC(GeoObjectOCC const &g) : GeoObjectOCC() {
    m_pimpl_->m_occ_shape_ = dynamic_cast<GeoObjectOCC const &>(g).m_pimpl_->m_occ_shape_;
    Update();
};
GeoObjectOCC::GeoObjectOCC(TopoDS_Shape const &shape) : GeoObjectOCC() {
    m_pimpl_->m_occ_shape_ = shape;
    Update();
}

GeoObjectOCC::~GeoObjectOCC(){};

TopoDS_Shape const &GeoObjectOCC::GetShape() const { return m_pimpl_->m_occ_shape_; }
Bnd_Box const &GeoObjectOCC::GetOCCBoundingBox() const { return m_pimpl_->m_occ_box_; }

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

void GeoObjectOCC::Transform(Real scale, point_type const &location, nTuple<Real, 4> const &rotate) {
    TopoDS_Shape tmp = m_pimpl_->m_occ_shape_;
    m_pimpl_->m_occ_shape_ = TransformShape(tmp, scale, location, rotate);
}
std::shared_ptr<data::DataTable> GeoObjectOCC::Serialize() const {
    auto res = GeoObject::Serialize();

    return res;
};
void GeoObjectOCC::Deserialize(std::shared_ptr<data::DataTable> const &cfg) {
    GeoObject::Deserialize(cfg);

    m_pimpl_->m_occ_shape_ =
        TransformShape(LoadShape(cfg->GetValue<std::string>("File", "")),
                       cfg->GetValue<Real>("Scale", 1.0e-3),  // default length unit is "m", STEP length unit is "mm"
                       cfg->GetValue("Location", point_type{0, 0, 0}),  //
                       cfg->GetValue("Rotation", nTuple<Real, 4>{0, 0, 0, 0}));

    Update();
    VERBOSE << " [ Bounding Box :" << m_pimpl_->m_bounding_box_ << "]" << std::endl;
};

void GeoObjectOCC::Load(std::string const &file_name) { m_pimpl_->m_occ_shape_ = LoadShape(file_name); };
void GeoObjectOCC::DoUpdate() {
    BRepBndLib::Add(m_pimpl_->m_occ_shape_, m_pimpl_->m_occ_box_);
    m_pimpl_->m_occ_box_.Get(std::get<0>(m_pimpl_->m_bounding_box_)[0], std::get<0>(m_pimpl_->m_bounding_box_)[1],
                             std::get<0>(m_pimpl_->m_bounding_box_)[2], std::get<1>(m_pimpl_->m_bounding_box_)[0],
                             std::get<1>(m_pimpl_->m_bounding_box_)[1], std::get<1>(m_pimpl_->m_bounding_box_)[2]);
}

box_type GeoObjectOCC::BoundingBox() const { return m_pimpl_->m_bounding_box_; };

bool GeoObjectOCC::CheckInside(point_type const &x) const {
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