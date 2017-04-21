//
// Created by salmon on 17-4-11.
//

#ifndef SIMPLA_EBMESH_H
#define SIMPLA_EBMESH_H
namespace simpla {
namespace mesh {
template <int NDIMS>
class EBMesh : public Mesh {
    Field<this_type, Real, VOLUME, 3> m_tags_{this};
};
}
}
#endif  // SIMPLA_EBMESH_H
