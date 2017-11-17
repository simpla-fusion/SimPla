//
// Created by salmon on 17-7-12.
//

#ifndef SIMPLA_EBMESH_H
#define SIMPLA_EBMESH_H

#include "simpla/SIMPLA_config.h"

#include <simpla/algebra/Algebra.h>
#include <simpla/algebra/EntityId.h>
#include <simpla/data/Data.h>
#include <simpla/engine/Attribute.h>
#include <simpla/engine/Domain.h>
#include <simpla/geometry/CutCell.h>
#include <simpla/utilities/Range.h>

namespace simpla {
namespace mesh {
using namespace data;
template <typename THost>
struct EBMesh {
    SP_DOMAIN_POLICY_HEAD(EBMesh);

    std::map<id_type, Real> m_volume_;
    std::map<id_type, Real> m_dual_volume_;

   public:
    void SetEmbeddedBoundary(std::string const &prefix, const std::shared_ptr<geometry::GeoObject> &g);

    engine::AttributeT<Real, NODE> m_vertex_tag_{m_host_, "Name"_ = "vertex_tag"};
    //    Field<host_type, Real, EDGE> m_edge_tag_{m_domain_, "name"_ = "edge_tag"};
    //    Field<host_type, Real, NODE, 3> m_edge_tag_d_{m_domain_, "name"_ = "edge_tag_d"};
    //    Field<host_type, Real, FACE> m_face_tag_{m_domain_, "name"_ = "face_tag"};
    engine::AttributeT<Real, CELL> m_volume_tag_{m_host_, "Name"_ = "volume_tag"};
};
template <typename THost>
EBMesh<THost>::EBMesh(THost *h) : m_host_(h) {}
template <typename THost>
EBMesh<THost>::~EBMesh() {}
template <typename THost>
std::shared_ptr<data::DataEntry> EBMesh<THost>::Serialize() const {
    return nullptr;
}
template <typename THost>
void EBMesh<THost>::Deserialize(std::shared_ptr<data::DataEntry> const &cfg) {}
template <typename THost>
void EBMesh<THost>::SetEmbeddedBoundary(std::string const &prefix, const std::shared_ptr<geometry::GeoObject> &g) {
    //    if (g == nullptr) { return; }
    //
    //    VERBOSE << "AddEntity Embedded Boundary [" << prefix << "]"
    //            << "Patch : Level=" << this->GetMesh()->GetMeshBlock()->GetLevel() << " "
    //            << this->GetMesh()->GetBoundingIndexBox(NODE) << std::endl;
    //
    //    Range<EntityId> body_ranges[4];
    //    Range<EntityId> boundary_ranges[4];
    //    std::map<EntityId, Real> cut_cell[4];
    //
    //    m_vertex_tag_.Clear();
    //    m_edge_tag_.Clear();

    //    geometry::CutCell(m_domain_->GetMesh()->GetChart(), m_domain_->GetMesh()->GetBoundingIndexBox(0b0), g,
    //    &m_volume_tag_.Get()[0]);
    //    m_edge_tag_d_[0] = m_edge_tag_.GetEntity();
    //    Real ratio = std::get<0>(m_domain_->GetMesh()->isOverlapped(g));
    //    if (ratio < EPSILON) {
    //    } else if (ratio < 1 - EPSILON) {
    //        detail::CreateEBMesh(m_domain_, prefix, g);
    //    } else {
    //        m_domain_->GetRange(prefix + "_BODY_0").SetFull();
    //        m_domain_->GetRange(prefix + "_BODY_1").SetFull();
    //        m_domain_->GetRange(prefix + "_BODY_2").SetFull();
    //        m_domain_->GetRange(prefix + "_BODY_3").SetFull();
    //    }
}
}  // namespace mesh
}  // namespace simpla
#endif  // SIMPLA_EBMESH_H
