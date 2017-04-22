//
// Created by salmon on 17-4-11.
//

#ifndef SIMPLA_RECTMESH_H
#define SIMPLA_RECTMESH_H

#include <simpla/engine/Mesh.h>

namespace simpla {
namespace mesh {

class RectMesh : public engine::Mesh {
    SP_OBJECT_HEAD(RectMesh, engine::Mesh)
   public:
    template <typename... Args>
    explicit RectMesh(Args &&... args) : engine::Mesh(std::forward<Args>(args)...){};
    ~RectMesh() override = default;

    SP_DEFAULT_CONSTRUCT(RectMesh)

    typedef EntityIdCoder M;
    point_type point(index_type i, index_type j, index_type k) const override = 0;

    point_type point(EntityId s) const override {
        return point(s, point_type{M::m_id_to_coordinates_shift_[s.w & 7][0], M::m_id_to_coordinates_shift_[s.w & 7][1],
                                   M::m_id_to_coordinates_shift_[s.w & 7][2]});
    }

    /**
    *\verbatim
    *                ^s (dl)
    *               /
    *   (dz) t     /
    *        ^    /
    *        |  110-------------111
    *        |  /|              /|
    *        | / |             / |
    *        |/  |            /  |
    *       100--|----------101  |
    *        | m |           |   |
    *        |  010----------|--011
    *        |  /            |  /
    *        | /             | /
    *        |/              |/
    *       000-------------001---> r (dr)
    *
    *\endverbatim
    */
    point_type point(EntityId id, point_type const &pr) const override {
        Real r = pr[0], s = pr[1], t = pr[2];

        Real w0 = (1 - r) * (1 - s) * (1 - t);
        Real w1 = r * (1 - s) * (1 - t);
        Real w2 = (1 - r) * s * (1 - t);
        Real w3 = r * s * (1 - t);
        Real w4 = (1 - r) * (1 - s) * t;
        Real w5 = r * (1 - s) * t;
        Real w6 = (1 - r) * s * t;
        Real w7 = r * s * t;
        point_type res;
        res = point(id.x /**/, id.y /**/, id.z /**/) * w0 + point(id.x + 1, id.y, id.z) * w1 +
              point(id.x /**/, id.y + 1, id.z /* */) * w2 + point(id.x + 1, id.y + 1, id.z) * w3 +
              point(id.x /**/, id.y /* */, id.z + 1) * w4 + point(id.x + 1, id.y, id.z + 1) * w5 +
              point(id.x /*  */, id.y + 1, id.z + 1) * w6 + point(id.x + 1, id.y + 1, id.z + 1) * w7;

        return res;
    }
};

}  // namespace mesh {
}  // namespace simpla {

#endif  // SIMPLA_RECTMESH_H
