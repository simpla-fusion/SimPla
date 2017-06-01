//
// Created by salmon on 17-4-25.
//
#include "CylindricalCoordinates.h"
#include <simpla/mesh/SMesh.h>
namespace simpla {
namespace geometry {

void CylindricalCoordinates::InitializeData(engine::MeshBase *m, Real time_now) const {
    auto *smesh = dynamic_cast<mesh::SMesh *>(m);
    smesh->InitializeData(time_now);

    smesh->m_coordinates_.Clear();

    /**
        *\verbatim
        *                ^y (dl)
        *               /
        *   (dz) z     /
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
        *       000-------------001---> x (dr)
        *
        *\endverbatim
        */

    point_type m_dx_ = GetScale();

    int Phi_axe = 2;  // std::dynamic_pointer_cast<CylindricalGeometry>(GetChart())->GetPhiAxe();
    int R_axe = (Phi_axe + 1) % 3;
    int Z_axe = (Phi_axe + 2) % 3;

    smesh->m_vertices_ = [&](EntityId s) -> Vec3 {
        return map(point_type{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)});
    };

    smesh->m_coordinates_ = [&](EntityId s) -> Vec3 {
        point_type x = map(point_type{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)});
        return Vec3{x[R_axe] * std::cos(x[Phi_axe]), x[R_axe] * std::sin(x[Phi_axe]), x[Z_axe]};
    };

    smesh->m_vertex_volume_ = 1.0;
    smesh->m_vertex_inv_volume_ = 1.0;
    smesh->m_vertex_dual_volume_ = [&](EntityId s) {
        point_type x = map(point_type{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)});
        Real dr = m_dx_[R_axe];
        Real dl0 = m_dx_[Phi_axe] * (x[R_axe] - 0.5 * m_dx_[R_axe]);
        Real dl1 = m_dx_[Phi_axe] * (x[R_axe] + 0.5 * m_dx_[R_axe]);
        Real dz = m_dx_[Z_axe];
        return 0.5 * dr * (dl0 + dl1) * dz;
    };
    smesh->m_vertex_inv_dual_volume_ = 1.0 / smesh->m_vertex_dual_volume_;

    smesh->m_volume_volume_ = [&](EntityId s) {
        point_type x = map(point_type{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)});
        Real dr = m_dx_[R_axe];
        Real dl0 = m_dx_[Phi_axe] * x[R_axe];
        Real dl1 = m_dx_[Phi_axe] * (x[R_axe] + m_dx_[R_axe]);
        Real dz = m_dx_[Z_axe];
        return 0.5 * dr * (dl0 + dl1) * dz;
    };
    smesh->m_volume_inv_volume_ = 1.0 / smesh->m_volume_volume_;
    smesh->m_volume_dual_volume_ = 1.0;
    smesh->m_volume_inv_dual_volume_ = 1.0;

    smesh->m_edge_volume_ = [&](EntityId s) -> Vec3 {
        point_type x = map(point_type{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)});
        Real dr = m_dx_[R_axe];
        Real dl0 = m_dx_[Phi_axe] * (x[R_axe]);
        Real dz = m_dx_[Z_axe];
        return {dr, dz, dl0};
    };
    smesh->m_edge_inv_volume_ = 1.0 / smesh->m_edge_volume_;
    smesh->m_edge_dual_volume_ = [&](EntityId s) -> Vec3 {
        point_type x = map(point_type{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)});
        Real dr = m_dx_[R_axe];
        Real dl0 = m_dx_[Phi_axe] * (x[R_axe] - 0.5 * m_dx_[R_axe]);
        Real dl1 = m_dx_[Phi_axe] * (x[R_axe] + 0.5 * m_dx_[R_axe]);
        Real dz = m_dx_[Z_axe];
        return {dl1 * dz, dr * (dl0 + dl1) * 0.5, dr * dz};
    };
    smesh->m_edge_inv_dual_volume_ = 1.0 / smesh->m_edge_dual_volume_;

    smesh->m_face_volume_ = [&](EntityId s) -> Vec3 {
        point_type x = map(point_type{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)});
        Real dr = m_dx_[R_axe];
        Real dl0 = m_dx_[Phi_axe] * (x[R_axe]);
        Real dl1 = m_dx_[Phi_axe] * (x[R_axe] + m_dx_[R_axe]);
        Real dz = m_dx_[Z_axe];
        return {dl0 * dz, dr * (dl0 + dl1) * 0.5, dr * dz};
    };

    smesh->m_face_inv_volume_ = 1.0 / smesh->m_face_volume_;
    smesh->m_face_dual_volume_ = [&](EntityId s) -> Vec3 {
        point_type x = map(point_type{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)});
        Real dr = m_dx_[R_axe];
        Real dl = m_dx_[Phi_axe] * (x[R_axe] + 0.5 * m_dx_[R_axe]);
        Real dz = m_dx_[Z_axe];
        return {dr, dz, dl};
    };
    smesh->m_face_inv_dual_volume_ = 1.0 / smesh->m_face_dual_volume_;
}

//    virtual point_type point(EntityId s) const override {
//        return GetChart()->inv_map(
//            point_type{static_cast<double>(s.x), static_cast<double>(s.y), static_cast<double>(s.z)});
//    };
//    virtual point_type point(EntityId id, point_type const &pr) const {

//
//        Real r = pr[0], s = pr[1], t = pr[2];
//
//        Real w0 = (1 - r) * (1 - s) * (1 - t);
//        Real w1 = r * (1 - s) * (1 - t);
//        Real w2 = (1 - r) * s * (1 - t);
//        Real w3 = r * s * (1 - t);
//        Real w4 = (1 - r) * (1 - s) * t;
//        Real w5 = r * (1 - s) * t;
//        Real w6 = (1 - r) * s * t;
//        Real w7 = r * s * t;
//
//        Real x =
//            m_vertices_(id.x /**/, id.y /**/, id.z /**/, 0) * w0 + m_vertices_(id.x + 1, id.y /**/, id.z /**/, 0) *
//            w1 +
//            m_vertices_(id.x /**/, id.y + 1, id.z /**/, 0) * w2 + m_vertices_(id.x + 1, id.y + 1, id.z /**/, 0) * w3
//            +
//            m_vertices_(id.x /**/, id.y /**/, id.z + 1, 0) * w4 + m_vertices_(id.x + 1, id.y /**/, id.z + 1, 0) * w5
//            +
//            m_vertices_(id.x /**/, id.y + 1, id.z + 1, 0) * w6 + m_vertices_(id.x + 1, id.y + 1, id.z + 1, 0) * w7;
//
//        Real y =
//            m_vertices_(id.x /**/, id.y /**/, id.z /**/, 1) * w0 + m_vertices_(id.x + 1, id.y /**/, id.z /**/, 1) *
//            w1 +
//            m_vertices_(id.x /**/, id.y + 1, id.z /**/, 1) * w2 + m_vertices_(id.x + 1, id.y + 1, id.z /**/, 1) * w3
//            +
//            m_vertices_(id.x /**/, id.y /**/, id.z + 1, 1) * w4 + m_vertices_(id.x + 1, id.y /**/, id.z + 1, 1) * w5
//            +
//            m_vertices_(id.x /**/, id.y + 1, id.z + 1, 1) * w6 + m_vertices_(id.x + 1, id.y + 1, id.z + 1, 1) * w7;
//
//        Real z =
//            m_vertices_(id.x /**/, id.y /**/, id.z /**/, 2) * w0 + m_vertices_(id.x + 1, id.y /**/, id.z /**/, 2) *
//            w1 +
//            m_vertices_(id.x /**/, id.y + 1, id.z /**/, 2) * w2 + m_vertices_(id.x + 1, id.y + 1, id.z /**/, 2) * w3
//            +
//            m_vertices_(id.x /**/, id.y /**/, id.z + 1, 2) * w4 + m_vertices_(id.x + 1, id.y /**/, id.z + 1, 2) * w5
//            +
//            m_vertices_(id.x /**/, id.y + 1, id.z + 1, 2) * w6 + m_vertices_(id.x + 1, id.y + 1, id.z + 1, 2) * w7;
//
//        return point_type{x, y, z};
//    }
}  // namespace mesh{
}  // namespace simpla{
