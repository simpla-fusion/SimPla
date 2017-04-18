//
// Created by salmon on 17-4-11.
//

#ifndef SIMPLA_CORECTMESH_H
#define SIMPLA_CORECTMESH_H

#include <iomanip>
#include <vector>
#include "simpla/engine/SPObjectHead.h"
#include "simpla/engine/all.h"
#include "simpla/geometry/Cube.h"
#include "simpla/mesh/EntityId.h"
#include "simpla/mesh/MeshCommon.h"
namespace simpla {
namespace mesh {

struct CoRectMesh : public engine::Mesh {
    SP_OBJECT_HEAD(CoRectMesh, engine::Mesh)
   public:
    static constexpr unsigned int NDIMS = 3;
    typedef Real scalar_type;
    typedef EntityId entity_id;

    CoRectMesh() : engine::Mesh(std::shared_ptr<engine::Chart>()) {}
    virtual ~CoRectMesh() {}
    this_type *Clone() const { return new this_type(*this); }
    void Initialize();

   private:
    std::shared_ptr<engine::Chart> m_chart_;
    std::shared_ptr<engine::MeshBlock> m_block_;
    nTuple<Real, 3> m_dx_, m_inv_dx_, m_x0_;
    Real m_v_[9];
    Real m_inv_v_[9];
    Real m_dual_v_[9];
    Real m_inv_dual_v_[9];

   public:
    point_type vertex(index_tuple const &x) const {
        return point_type{m_x0_[0] + m_dx_[0] * x[0], m_x0_[1] + m_dx_[1] * x[1], m_x0_[2] + m_dx_[2] * x[2]};
    }

    std::pair<index_tuple, point_type> map(point_type const &x) const {
        point_type r;
        r = (x - m_x0_) * m_inv_dx_;
        index_tuple idx;
        for (int i = 0; i < 3; ++i) {
            idx[i] = static_cast<index_type>(r[i]);
            r[i] -= idx[i];
        }
        return std::make_pair(idx, r);
    }

    point_type inv_map(std::pair<index_tuple, point_type> const &p) const {
        point_type r;
        r = (p.first + p.second) * m_dx_ + m_x0_;
        index_tuple idx;
        for (int i = 0; i < 3; ++i) {
            idx[i] = static_cast<index_type>(r[i]);
            r[i] -= idx[i];
        }
        return r;
    }

    Real volume(index_tuple const &x, int d = 0) const { return m_v_[d]; }
    Real dual_volume(index_tuple const &x, int d = 0) const { return m_dual_v_[d]; }
    Real inv_volume(index_tuple const &x, int d = 0) const { return m_inv_v_[d]; }
    Real inv_dual_volume(index_tuple const &x, int d = 0) const { return m_inv_dual_v_[d]; }

};  // struct  Mesh

// template <>
// struct mesh_traits<CoRectMesh> {
//    typedef CoRectMesh type;
//    typedef entity_id entity_id;
//    typedef Real scalar_type;
//
//    template <int IFORM, int DOF>
//    struct Shift {
//        template <typename... Args>
//        Shift(Args &&... args) {}
//        constexpr entity_id operator()(entity_id const &s) const { return s; }
//    };
//};

inline void CoRectMesh::Initialize() {
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
    m_x0_ = GetChart()->GetOrigin();
    m_dx_ = GetChart()->GetDx();
    size_tuple m_dims_ = GetBlock()->GetDimensions();

    m_v_[0 /*000*/] = 1;
    m_v_[1 /*001*/] = (m_dims_[0] == 1) ? 1 : m_dx_[0];
    m_v_[2 /*010*/] = (m_dims_[1] == 1) ? 1 : m_dx_[1];
    m_v_[4 /*100*/] = (m_dims_[2] == 1) ? 1 : m_dx_[2];
    m_v_[3 /*011*/] = m_v_[1] * m_v_[2];
    m_v_[5 /*101*/] = m_v_[4] * m_v_[1];
    m_v_[6 /*110*/] = m_v_[4] * m_v_[2];
    m_v_[7 /*111*/] = m_v_[1] * m_v_[2] * m_v_[4];

    m_dual_v_[0 /*000*/] = m_v_[7];
    m_dual_v_[1 /*001*/] = m_v_[6];
    m_dual_v_[2 /*010*/] = m_v_[5];
    m_dual_v_[4 /*100*/] = m_v_[3];
    m_dual_v_[3 /*011*/] = m_v_[4];
    m_dual_v_[5 /*101*/] = m_v_[2];
    m_dual_v_[6 /*110*/] = m_v_[1];
    m_dual_v_[7 /*111*/] = m_v_[0];

    m_inv_v_[0 /*000*/] = 1;
    m_inv_v_[1 /*001*/] = (m_dims_[0] == 1) ? 1 : m_inv_dx_[0];
    m_inv_v_[2 /*010*/] = (m_dims_[1] == 1) ? 1 : m_inv_dx_[1];
    m_inv_v_[4 /*100*/] = (m_dims_[2] == 1) ? 1 : m_inv_dx_[2];
    m_inv_v_[3 /*011*/] = m_inv_v_[2] * m_inv_v_[1];
    m_inv_v_[5 /*101*/] = m_inv_v_[4] * m_inv_v_[1];
    m_inv_v_[6 /*110*/] = m_inv_v_[4] * m_inv_v_[2];
    m_inv_v_[7 /*111*/] = m_inv_v_[1] * m_inv_v_[2] * m_inv_v_[4];

    m_inv_v_[1 /*001*/] = (m_dims_[0] == 1) ? 0 : m_inv_v_[1];
    m_inv_v_[2 /*010*/] = (m_dims_[1] == 1) ? 0 : m_inv_v_[2];
    m_inv_v_[4 /*100*/] = (m_dims_[2] == 1) ? 0 : m_inv_v_[4];

    m_inv_dual_v_[0 /*000*/] = m_inv_v_[7];
    m_inv_dual_v_[1 /*001*/] = m_inv_v_[6];
    m_inv_dual_v_[2 /*010*/] = m_inv_v_[5];
    m_inv_dual_v_[4 /*100*/] = m_inv_v_[3];
    m_inv_dual_v_[3 /*011*/] = m_inv_v_[4];
    m_inv_dual_v_[5 /*101*/] = m_inv_v_[2];
    m_inv_dual_v_[6 /*110*/] = m_inv_v_[1];
    m_inv_dual_v_[7 /*111*/] = m_inv_v_[0];
}
}  // namespace  mesh
}  // namespace simpla

#endif  // SIMPLA_CORECTMESH_H
