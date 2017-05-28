//
// Created by salmon on 17-4-11.
//

#ifndef SIMPLA_CORECTMESH_H
#define SIMPLA_CORECTMESH_H

#include <simpla/utilities/sp_def.h>
#include <string>
#include "StructuredMesh.h"
namespace simpla {
namespace mesh {

struct CoRectMesh : public StructuredMesh {
    SP_OBJECT_HEAD(CoRectMesh, StructuredMesh)

   public:
    template <typename... Args>
    explicit CoRectMesh(Args &&... args) : StructuredMesh(std::forward<Args>(args)...){};
    ~CoRectMesh() override = default;

    static std::string ClassName() { return std::string("CoRectMesh"); }
    SP_DEFAULT_CONSTRUCT(CoRectMesh)

    void InitialCondition(Real time_now) override;

    nTuple<Real, 3> m_dx_{1, 1, 1}, m_inv_dx_{1, 1, 1}, m_x0_{0, 0, 0};

    Real m_vertex_volume_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_vertex_inv_volume_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_vertex_dual_volume_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_vertex_inv_dual_volume_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_volume_volume_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_volume_inv_volume_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_volume_dual_volume_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_volume_inv_dual_volume_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_edge_volume_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_edge_inv_volume_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_edge_dual_volume_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_edge_inv_dual_volume_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_face_volume_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_face_inv_volume_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_face_dual_volume_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_face_inv_dual_volume_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_vertex_hodge_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_edge_hodge_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_face_hodge_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
    Real m_volume_hodge_[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};

    using StructuredMesh::point;

    point_type point(index_type x, index_type y, index_type z) const override {
        return point_type{m_x0_[0] + m_dx_[0] * x, m_x0_[1] + m_dx_[1] * y, m_x0_[2] + m_dx_[2] * z};
    }

    //    Real volume(EntityId s) const override { return m_volume_[s.w & 7]; }
    //    Real dual_volume(EntityId s) const override { return m_volume_[s.w & 7]; }
    //    Real inv_volume(EntityId s) const override { return m_volume_[s.w & 7]; }
    //    Real inv_dual_volume(EntityId s) const override { return m_volume_[s.w & 7]; }

    std::pair<EntityId, point_type> map(point_type const &x, int node_id = VERTEX) const {
        point_type r;

        r[0] = std::fma(x[0] - m_x0_[0], m_inv_dx_[0], m_x0_[0] * m_inv_dx_[0]);
        r[1] = std::fma(x[1] - m_x0_[1], m_inv_dx_[1], m_x0_[1] * m_inv_dx_[1]);
        r[2] = std::fma(x[2] - m_x0_[2], m_inv_dx_[2], m_x0_[2] * m_inv_dx_[2]);

        EntityId s;
        index_tuple idx;
        idx = r;
        r -= idx;
        s.w = static_cast<int16_t>(node_id);
        s.x = static_cast<int16_t>(r[0]);
        s.y = static_cast<int16_t>(r[1]);
        s.z = static_cast<int16_t>(r[2]);

        r[0] -= s.x;
        r[1] -= s.y;
        r[2] -= s.z;
        return std::make_pair(s, r);
    }

    point_type inv_map(std::pair<EntityId, point_type> const &p) const {
        point_type r;

        r[0] = std::fma(static_cast<Real>(p.first.x + p.second[0]), m_dx_[0], m_x0_[0]);
        r[1] = std::fma(static_cast<Real>(p.first.y + p.second[1]), m_dx_[1], m_x0_[1]);
        r[2] = std::fma(static_cast<Real>(p.first.z + p.second[2]), m_dx_[2], m_x0_[2]);

        index_tuple idx;
        for (int i = 0; i < 3; ++i) {
            idx[i] = static_cast<index_type>(r[i]);
            r[i] -= idx[i];
        }
        return r;
    }

};  // struct  MeshBase

inline void CoRectMesh::InitialCondition(Real time_now) {
    //    StructuredMesh::InitialCondition(time_now);
    /**
        *\verbatim
        *                ^y
        *               /
        *        z     /
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
        *       000-------------001---> x
        *
        *\endverbatim
        */
    //    m_x0_ = GetChart()->GetOrigin();
    //    m_dx_ = GetChart()->GetDx();
    size_tuple m_dims_ = GetBlock()->GetDimensions();

//    m_volume_[0 /*000*/] = 1;
//    m_volume_[1 /*001*/] = (m_dims_[0] == 1) ? 1 : m_dx_[0];
//    m_volume_[2 /*010*/] = (m_dims_[1] == 1) ? 1 : m_dx_[1];
//    m_volume_[4 /*100*/] = (m_dims_[2] == 1) ? 1 : m_dx_[2];
//    m_volume_[3 /*011*/] = m_volume_[1] * m_volume_[2];
//    m_volume_[5 /*101*/] = m_volume_[4] * m_volume_[1];
//    m_volume_[6 /*110*/] = m_volume_[4] * m_volume_[2];
//    m_volume_[7 /*111*/] = m_volume_[1] * m_volume_[2] * m_volume_[4];
//
//    m_dual_volume_[0 /*000*/] = m_volume_[7];
//    m_dual_volume_[1 /*001*/] = m_volume_[6];
//    m_dual_volume_[2 /*010*/] = m_volume_[5];
//    m_dual_volume_[4 /*100*/] = m_volume_[3];
//    m_dual_volume_[3 /*011*/] = m_volume_[4];
//    m_dual_volume_[5 /*101*/] = m_volume_[2];
//    m_dual_volume_[6 /*110*/] = m_volume_[1];
//    m_dual_volume_[7 /*111*/] = m_volume_[0];
//
//    m_inv_volume_[0 /*000*/] = 1;
//    m_inv_volume_[1 /*001*/] = (m_dims_[0] == 1) ? 1 : m_inv_dx_[0];
//    m_inv_volume_[2 /*010*/] = (m_dims_[1] == 1) ? 1 : m_inv_dx_[1];
//    m_inv_volume_[4 /*100*/] = (m_dims_[2] == 1) ? 1 : m_inv_dx_[2];
//    m_inv_volume_[3 /*011*/] = m_inv_volume_[2] * m_inv_volume_[1];
//    m_inv_volume_[5 /*101*/] = m_inv_volume_[4] * m_inv_volume_[1];
//    m_inv_volume_[6 /*110*/] = m_inv_volume_[4] * m_inv_volume_[2];
//    m_inv_volume_[7 /*111*/] = m_inv_volume_[1] * m_inv_volume_[2] * m_inv_volume_[4];
//
//    m_inv_volume_[1 /*001*/] = (m_dims_[0] == 1) ? 0 : m_inv_volume_[1];
//    m_inv_volume_[2 /*010*/] = (m_dims_[1] == 1) ? 0 : m_inv_volume_[2];
//    m_inv_volume_[4 /*100*/] = (m_dims_[2] == 1) ? 0 : m_inv_volume_[4];
//
//    m_inv_dual_volume_[0 /*000*/] = m_inv_volume_[7];
//    m_inv_dual_volume_[1 /*001*/] = m_inv_volume_[6];
//    m_inv_dual_volume_[2 /*010*/] = m_inv_volume_[5];
//    m_inv_dual_volume_[4 /*100*/] = m_inv_volume_[3];
//    m_inv_dual_volume_[3 /*011*/] = m_inv_volume_[4];
//    m_inv_dual_volume_[5 /*101*/] = m_inv_volume_[2];
//    m_inv_dual_volume_[6 /*110*/] = m_inv_volume_[1];
//    m_inv_dual_volume_[7 /*111*/] = m_inv_volume_[0];
}
}  // namespace  mesh
}  // namespace simpla

#endif  // SIMPLA_CORECTMESH_H
