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
#include "simpla/utilities/Range.h"

namespace simpla {
namespace mesh {

template <typename THost>
struct EBMesh {
    SP_ENGINE_POLICY_HEAD(EBMesh);

    std::map<id_type, Real> m_volume_;
    std::map<id_type, Real> m_dual_volume_;

   public:
    void SetEmbeddedBoundary(std::string const &prefix, const geometry::GeoObject *g);
};

namespace detail {
void CreateEBMesh(engine::MeshBase *m_host_, std::string const &prefix, geometry::GeoObject const *g);
}
template <typename THost>
void EBMesh<THost>::SetEmbeddedBoundary(std::string const &prefix, const geometry::GeoObject *g) {
    if (g == nullptr) { return; }

    VERBOSE << "Add Embedded Boundary [" << prefix << "]"
            << "Patch : Level=" << m_host_->GetMesh()->GetBlock()->GetLevel() << " " << m_host_->GetMesh()->IndexBox(0)
            << std::endl;

    Real ratio = std::get<0>(m_host_->GetMesh()->CheckOverlap(g));
    if (ratio < EPSILON) {
    } else if (ratio < 1 - EPSILON) {
        detail::CreateEBMesh(m_host_, prefix, g);
    } else {
        m_host_->GetRange(prefix + "_BODY_0").SetFull();
        m_host_->GetRange(prefix + "_BODY_1").SetFull();
        m_host_->GetRange(prefix + "_BODY_2").SetFull();
        m_host_->GetRange(prefix + "_BODY_3").SetFull();
    }
}
}  // namespace mesh
}  // namespace simpla
#endif  // SIMPLA_EBMESH_H
