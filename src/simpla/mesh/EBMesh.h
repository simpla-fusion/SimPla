//
// Created by salmon on 17-4-22.
//

#ifndef SIMPLA_EBMESH_H
#define SIMPLA_EBMESH_H

namespace simpla {
namespace mesh {
template <typename BMesh>
class EBMesh : public BMesh {
    virtual Real volume(EntityId s) const { return m_volume_[s.w & 7](s.x, s.y, s.z); }
    virtual Real dual_volume(EntityId s) const { return m_volume_[s.w & 7](s.x, s.y, s.z); }
    virtual Real inv_volume(EntityId s) const { return m_volume_[s.w & 7](s.x, s.y, s.z); }
    virtual Real inv_dual_volume(EntityId s) const { return m_volume_[s.w & 7](s.x, s.y, s.z); }
};
}
}

#endif  // SIMPLA_EBMESH_H
