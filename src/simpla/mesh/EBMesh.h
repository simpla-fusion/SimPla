//
// Created by salmon on 17-7-12.
//

#ifndef SIMPLA_EBMESH_H
#define SIMPLA_EBMESH_H

#include "simpla/SIMPLA_config.h"

#include "simpla/algebra/Algebra.h"
#include "simpla/algebra/EntityId.h"
#include "simpla/data/Data.h"
#include "simpla/engine/Domain.h"
#include "simpla/geometry/CutCell.h"
#include "simpla/utilities/Range.h"

namespace simpla {
namespace mesh {
using namespace data;
template <typename THost>
struct EBMesh {
    SP_ENGINE_POLICY_HEAD(EBMesh);

    std::map<id_type, Real> m_volume_;
    std::map<id_type, Real> m_dual_volume_;

   public:
    void SetEmbeddedBoundary(std::string const &prefix, const std::shared_ptr<geometry::GeoObject> &g);

    Field<host_type, Real, NODE> m_vertex_tag_{m_host_, "name"_ = "vertex_tag"};
    //    Field<host_type, Real, EDGE> m_edge_tag_{m_host_, "name"_ = "edge_tag"};
    //    Field<host_type, Real, NODE, 3> m_edge_tag_d_{m_host_, "name"_ = "edge_tag_d"};
    //    Field<host_type, Real, FACE> m_face_tag_{m_host_, "name"_ = "face_tag"};
    Field<host_type, Real, CELL> m_volume_tag_{m_host_, "name"_ = "volume_tag"};
};
template <typename THost>
std::shared_ptr<data::DataNode> EBMesh<THost>::Serialize() const {
    return nullptr;
}
template <typename THost>
void EBMesh<THost>::Deserialize(std::shared_ptr<data::DataNode> const &cfg) {}
template <typename THost>
void EBMesh<THost>::SetEmbeddedBoundary(std::string const &prefix, const std::shared_ptr<geometry::GeoObject> &g) {
    if (g == nullptr) { return; }

    VERBOSE << "AddEntity Embedded Boundary [" << prefix << "]"
            << "Patch : Level=" << m_host_->GetMesh()->GetBlock()->GetLevel() << " "
            << m_host_->GetMesh()->IndexBox(NODE) << std::endl;

    Range<EntityId> body_ranges[4];
    Range<EntityId> boundary_ranges[4];
    std::map<EntityId, Real> cut_cell[4];

    m_vertex_tag_.Clear();
    //    m_edge_tag_.Clear();

//    geometry::CutCell(m_host_->GetMesh()->GetChart(), m_host_->GetMesh()->IndexBox(0b0), g, &m_volume_tag_.Get()[0]);
    //    m_edge_tag_d_[0] = m_edge_tag_.GetEntity();
    //    Real ratio = std::get<0>(m_host_->GetMesh()->CheckOverlap(g));
    //    if (ratio < EPSILON) {
    //    } else if (ratio < 1 - EPSILON) {
    //        detail::CreateEBMesh(m_host_, prefix, g);
    //    } else {
    //        m_host_->GetRange(prefix + "_BODY_0").SetFull();
    //        m_host_->GetRange(prefix + "_BODY_1").SetFull();
    //        m_host_->GetRange(prefix + "_BODY_2").SetFull();
    //        m_host_->GetRange(prefix + "_BODY_3").SetFull();
    //    }
}
}  // namespace mesh
}  // namespace simpla
#endif  // SIMPLA_EBMESH_H
