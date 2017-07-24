//
// Created by salmon on 17-7-22.
//

#include "GeoObjectOCC.h"

#include <BRepAdaptor_Surface.hxx>
#include <BRepAlgoAPI_Section.hxx>
#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_MakeEdge.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <BRepTools.hxx>
#include <BRep_Builder.hxx>
#include <BinTools_ShapeSet.hxx>
#include <BndLib_AddSurface.hxx>
#include <Bnd_Box.hxx>
#include <Precision.hxx>
#include <STEPControl_Reader.hxx>
#include <Standard_Boolean.hxx>
#include <TCollection_ExtendedString.hxx>
#include <TDF_ChildIterator.hxx>
#include <TDF_Label.hxx>
#include <TDocStd_Document.hxx>
#include <TopExp_Explorer.hxx>
#include <TopTools_DataMapOfShapeInteger.hxx>
#include <TopoDS.hxx>
#include <TopoDS_CompSolid.hxx>
#include <TopoDS_Compound.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Shell.hxx>
#include <TopoDS_Solid.hxx>
#include <TopoDS_Vertex.hxx>
#include <TopoDS_Wire.hxx>

namespace simpla {
namespace geometry {
REGISTER_CREATOR(GeoObjectOCC, occ)

struct GeoObjectOCC::pimpl_s {
    std::string m_file_;
    std::string m_label_;

    std::shared_ptr<TopoDS_Shape> m_occ_shape_;
    box_type m_bound_box_{{0, 0, 0}, {0, 0, 0}};
};
GeoObjectOCC::GeoObjectOCC() : m_pimpl_(new pimpl_s){};
GeoObjectOCC::~GeoObjectOCC(){};

std::shared_ptr<data::DataTable> GeoObjectOCC::Serialize() const {
    auto res = GeoObject::Serialize();
    res->SetValue("File", m_pimpl_->m_file_);
    res->SetValue("Label", m_pimpl_->m_label_);

    return res;
};
void GeoObjectOCC::Deserialize(std::shared_ptr<data::DataTable> const &cfg) {
    GeoObject::Deserialize(cfg);
    Load(cfg->GetValue("File", m_pimpl_->m_file_), cfg->GetValue("Label", m_pimpl_->m_label_));

    Update();
};
void GeoObjectOCC::Load(std::string const &file_name, std::string const &label) {
    m_pimpl_->m_file_ = file_name;
    m_pimpl_->m_label_ = label;
    m_pimpl_->m_occ_shape_.reset(new TopoDS_Shape);

    STEPControl_Reader reader;
    if (reader.ReadFile(file_name.c_str()) != IFSelect_RetDone) {
        RUNTIME_ERROR << "Real STEP file failed!" << std::endl;
    } else {
        //        reader.TransferRoots();
        CHECK(reader.TransferList(reader.GiveList(label.c_str())));
        *m_pimpl_->m_occ_shape_ = reader.Shape();
        CHECK(m_pimpl_->m_occ_shape_->ShapeType());
    }
};
void GeoObjectOCC::DoUpdate() {
    Bnd_Box box;
    BRepBndLib::Add(*m_pimpl_->m_occ_shape_, box);
    box.Get(std::get<0>(m_pimpl_->m_bound_box_)[0], std::get<0>(m_pimpl_->m_bound_box_)[1],
            std::get<0>(m_pimpl_->m_bound_box_)[2], std::get<1>(m_pimpl_->m_bound_box_)[0],
            std::get<1>(m_pimpl_->m_bound_box_)[1], std::get<1>(m_pimpl_->m_bound_box_)[2]);
}
box_type GeoObjectOCC::BoundingBox() const { return m_pimpl_->m_bound_box_; };
bool GeoObjectOCC::CheckInside(point_type const &x) const {
    //    VERBOSE << m_pimpl_->m_bound_box_ << (x) << std::endl;
    gp_Pnt p(x[0], x[1], x[2]);
    //    gp_Pnt p(0,0,0);
    BRepBuilderAPI_MakeVertex vertex(p);
    BRepExtrema_DistShapeShape dist(vertex, *m_pimpl_->m_occ_shape_);
    dist.Perform();
    CHECK(dist.Value()) << dist.InnerSolution();
    return dist.InnerSolution();
};

}  // namespace geometry
}  // namespace simpla
