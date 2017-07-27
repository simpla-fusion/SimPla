//
// Created by salmon on 17-7-27.
//

#include "GeoObjectOCC.h"

#include <BRepBndLib.hxx>
#include <BRepBuilderAPI_MakeVertex.hxx>
#include <BRepExtrema_DistShapeShape.hxx>
#include <Bnd_Box.hxx>
#include <Interface_Static.hxx>
#include <STEPControl_Reader.hxx>
#include <TColStd_HSequenceOfTransient.hxx>
#include <TopoDS_Shape.hxx>
#include "simpla/utilities/SPDefines.h"

namespace simpla {
namespace geometry {
REGISTER_CREATOR(GeoObjectOCC, occ)

struct GeoObjectOCC::pimpl_s {
    std::string m_file_;
    std::string m_label_;
    Real m_measure_ = SNaN;
    std::shared_ptr<TopoDS_Shape> m_occ_shape_;
    box_type m_bounding_box_{{0, 0, 0}, {0, 0, 0}};
};
GeoObjectOCC::GeoObjectOCC() : m_pimpl_(new pimpl_s){};

GeoObjectOCC::GeoObjectOCC(GeoObject const &g) : GeoObjectOCC() {
    if (!g.isA(typeid(GeoObjectOCC))) {
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

TopoDS_Shape GeoObjectOCC::GetShape() const { return *m_pimpl_->m_occ_shape_; }

void GeoObjectOCC::Load(std::string const &file_name, std::string const &label) {
    m_pimpl_->m_file_ = file_name;
    m_pimpl_->m_label_ = label;
    m_pimpl_->m_occ_shape_.reset(new TopoDS_Shape);

    STEPControl_Reader reader;

    auto success = reader.ReadFile(file_name.c_str());
    if (!Interface_Static::SetIVal("xstep.cascade.unit", 1000)) {
        RUNTIME_ERROR << "Set Value xstep.cascade.unit fail!" << std::endl;
    };

    if (success != IFSelect_RetDone) {
        RUNTIME_ERROR << "Real STEP file failed!" << std::endl;
    } else {
        if (label.empty()) {
            reader.TransferRoots();
            *m_pimpl_->m_occ_shape_ = reader.OneShape();
        } else {
            int num = reader.TransferList(reader.GiveList(label.c_str()));

            if (num == 0) {
                OUT_OF_RANGE << "STEP object:" << file_name << ":" << label << " is not found! ["
                             << "] " << std::endl;
            }
            *m_pimpl_->m_occ_shape_ = reader.Shape();
        }
    }
    Update();
    VERBOSE << "STEP Object is loaded from " << file_name << " [ Bounding Box :" << m_pimpl_->m_bounding_box_ << "]"
            << std::endl;
};
void GeoObjectOCC::DoUpdate() {
    Bnd_Box box;
    BRepBndLib::Add(*m_pimpl_->m_occ_shape_, box);
    box.Get(std::get<0>(m_pimpl_->m_bounding_box_)[0], std::get<0>(m_pimpl_->m_bounding_box_)[1],
            std::get<0>(m_pimpl_->m_bounding_box_)[2], std::get<1>(m_pimpl_->m_bounding_box_)[0],
            std::get<1>(m_pimpl_->m_bounding_box_)[1], std::get<1>(m_pimpl_->m_bounding_box_)[2]);
}

box_type GeoObjectOCC::BoundingBox() const { return m_pimpl_->m_bounding_box_; };
bool GeoObjectOCC::CheckInside(point_type const &x) const {
    //    VERBOSE << m_pimpl_->m_bounding_box_ << (x) << std::endl;
    gp_Pnt p(x[0], x[1], x[2]);
    //    gp_Pnt p(0,0,0);
    BRepBuilderAPI_MakeVertex vertex(p);
    BRepExtrema_DistShapeShape dist(vertex, *m_pimpl_->m_occ_shape_);
    dist.Perform();
    CHECK(dist.Value()) << dist.InnerSolution();
    return dist.InnerSolution();
};
}
}  // namespace simpla