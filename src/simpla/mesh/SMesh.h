//
// Created by salmon on 17-4-24.
//

#ifndef SIMPLA_SMESH_H
#define SIMPLA_SMESH_H

#include <simpla/algebra/all.h>
#include <simpla/data/all.h>
#include "StructuredMesh.h"
namespace simpla {
namespace mesh {
using namespace simpla::engine;

struct SMesh : public StructuredMesh {
   public:
    SP_OBJECT_HEAD(SMesh, StructuredMesh)

    explicit SMesh(Domain *d) : StructuredMesh(d){};
    ~SMesh() override = default;

    SP_DEFAULT_CONSTRUCT(SMesh)
    DECLARE_REGISTER_NAME("SMesh");
    void InitializeData(Real time_now) override;

    Field<this_type, Real, VERTEX, 3> m_coordinates_{this, "COORDINATES"_, "name"_ = "Coordinates"};
    Field<this_type, Real, VERTEX, 3> m_vertices_{this /* , "name"_ = "vertices"*/};
    Field<this_type, Real, VOLUME, 9> m_volume_{this /* ,"name"_ = "volume"*/};
    Field<this_type, Real, VOLUME, 9> m_dual_volume_{this /* ,"name"_ = "dual_volume"*/};
    Field<this_type, Real, VOLUME, 9> m_inv_volume_{this /* , "name"_ = "inv_volume"*/};
    Field<this_type, Real, VOLUME, 9> m_inv_dual_volume_{this /* ,"name"_ = "inv_dual_volume"*/};

    point_type point(EntityId s) const override { return StructuredMesh::point(s); }

    Real volume(EntityId s) const override { return m_volume_[s]; }
    Real dual_volume(EntityId s) const override { return m_volume_[s]; }
    Real inv_volume(EntityId s) const override { return m_volume_[s]; }
    Real inv_dual_volume(EntityId s) const override { return m_volume_[s]; }
};

}  // namespace mesh {
}  // namespace simpla {
#endif  // SIMPLA_SMESH_H
