//
// Created by salmon on 17-4-25.
//
#include "CylindricalGeometry.h"

namespace simpla {
namespace mesh {
void CylindricalSMesh::InitializeData(Real time_now) {
    SMesh::InitializeData(time_now);
    m_coordinates_.Clear();

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

    point_type m_dx_ = GetChart()->GetScale();

    int Phi_axe = 2;  // std::dynamic_pointer_cast<CylindricalGeometry>(GetChart())->GetPhiAxe();
    int R_axe = (Phi_axe + 1) % 3;
    int Z_axe = (Phi_axe + 2) % 3;

    m_vertices_ = [&](EntityId s) -> Vec3 {
        return map(point_type{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)});
    };

    m_coordinates_ = [&](EntityId s) -> Vec3 {
        point_type x = map(point_type{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)});
        return Vec3{x[R_axe] * std::cos(x[Phi_axe]), x[R_axe] * std::sin(x[Phi_axe]), x[Z_axe]};
    };

    m_vertex_volume_ = 1.0;
    m_vertex_inv_volume_ = 1.0;
    m_vertex_dual_volume_ = [&](EntityId s) {
        point_type x = map(point_type{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)});
        Real dr = m_dx_[R_axe];
        Real dl0 = m_dx_[Phi_axe] * (x[R_axe] - 0.5 * m_dx_[R_axe]);
        Real dl1 = m_dx_[Phi_axe] * (x[R_axe] + 0.5 * m_dx_[R_axe]);
        Real dz = m_dx_[Z_axe];
        return 0.5 * dr * (dl0 + dl1) * dz;
    };
    m_vertex_inv_dual_volume_ = 1.0 / m_vertex_dual_volume_;

    m_volume_volume_ = [&](EntityId s) {
        point_type x = map(point_type{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)});
        Real dr = m_dx_[R_axe];
        Real dl0 = m_dx_[Phi_axe] * x[R_axe];
        Real dl1 = m_dx_[Phi_axe] * (x[R_axe] + m_dx_[R_axe]);
        Real dz = m_dx_[Z_axe];
        return 0.5 * dr * (dl0 + dl1) * dz;
    };
    m_volume_inv_volume_ = 1.0 / m_volume_volume_;
    m_volume_dual_volume_ = 1.0;
    m_volume_inv_dual_volume_ = 1.0;

    m_edge_volume_ = [&](EntityId s) -> Vec3 {
        point_type x = map(point_type{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)});
        Real dr = m_dx_[R_axe];
        Real dl0 = m_dx_[Phi_axe] * (x[R_axe]);
        Real dz = m_dx_[Z_axe];
        return {dr, dz, dl0};
    };
    m_edge_inv_volume_ = 1.0 / m_edge_volume_;
    m_edge_dual_volume_ = [&](EntityId s) -> Vec3 {
        point_type x = map(point_type{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)});
        Real dr = m_dx_[R_axe];
        Real dl0 = m_dx_[Phi_axe] * (x[R_axe] - 0.5 * m_dx_[R_axe]);
        Real dl1 = m_dx_[Phi_axe] * (x[R_axe] + 0.5 * m_dx_[R_axe]);
        Real dz = m_dx_[Z_axe];
        return {dl1 * dz, dr * (dl0 + dl1) * 0.5, dr * dz};
    };
    m_edge_inv_dual_volume_ = 1.0 / m_edge_dual_volume_;

    m_face_volume_ = [&](EntityId s) -> Vec3 {
        point_type x = map(point_type{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)});
        Real dr = m_dx_[R_axe];
        Real dl0 = m_dx_[Phi_axe] * (x[R_axe]);
        Real dl1 = m_dx_[Phi_axe] * (x[R_axe] + m_dx_[R_axe]);
        Real dz = m_dx_[Z_axe];
        return {dl0 * dz, dr * (dl0 + dl1) * 0.5, dr * dz};
    };

    m_face_inv_volume_ = 1.0 / m_face_volume_;
    m_face_dual_volume_ = [&](EntityId s) -> Vec3 {
        point_type x = map(point_type{static_cast<Real>(s.x), static_cast<Real>(s.y), static_cast<Real>(s.z)});
        Real dr = m_dx_[R_axe];
        Real dl = m_dx_[Phi_axe] * (x[R_axe] + 0.5 * m_dx_[R_axe]);
        Real dz = m_dx_[Z_axe];
        return {dr, dz, dl};
    };
    m_face_inv_dual_volume_ = 1.0 / m_face_dual_volume_;
}
}  // namespace mesh{
}  // namespace simpla{
