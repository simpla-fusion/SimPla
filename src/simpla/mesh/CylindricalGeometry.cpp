//
// Created by salmon on 17-4-25.
//
#include "CylindricalGeometry.h"
namespace simpla {
namespace mesh {
void Mesh<CylindricalGeometry, SMesh>::InitializeData(Real time_now) {
    SMesh::InitializeData(time_now);

    auto &m_vertics_ = GetVertics();
    auto &m_volume_ = GetVolume();
    auto &m_dual_volume_ = GetDualVolume();
    auto &m_inv_volume_ = GetInvVolume();
    auto &m_inv_dual_volume_ = GetInvDualVolume();

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
    auto const *lower = &(std::get<0>(m_vertics_[0].GetIndexBox())[0]);
    auto const *upper = &(std::get<1>(m_vertics_[0].GetIndexBox())[0]);

    index_type ib = lower[0];
    index_type ie = upper[0];
    index_type jb = lower[1];
    index_type je = upper[1];
    index_type kb = lower[2];
    index_type ke = upper[2];
    point_type m_dx_ = GetChart()->GetDx();
    point_type x0 = GetChart()->GetOrigin();

    for (index_type i = ib; i < ie; ++i)
        for (index_type j = jb; j < je; ++j)
            for (index_type k = kb; k < ke; ++k) {
                point_type x =
                    GetChart()->inv_map(point_type{static_cast<Real>(i), static_cast<Real>(j), static_cast<Real>(k)});
                m_vertics_[0](i, j, k) = x[0] * std::cos(x[1]);
                m_vertics_[1](i, j, k) = x[0] * std::sin(x[1]);
                m_vertics_[2](i, j, k) = x[2];
            }

    ib = std::get<0>(m_volume_[0].GetIndexBox())[0];
    jb = std::get<0>(m_volume_[0].GetIndexBox())[1];
    kb = std::get<0>(m_volume_[0].GetIndexBox())[2];
    ie = std::get<1>(m_volume_[0].GetIndexBox())[0];
    je = std::get<1>(m_volume_[0].GetIndexBox())[1];
    ke = std::get<1>(m_volume_[0].GetIndexBox())[2];

    for (index_type i = ib; i < ie; ++i)
        for (index_type j = jb; j < je; ++j)
            for (index_type k = kb; k < ke; ++k) {
                point_type x =
                    GetChart()->inv_map(point_type{static_cast<Real>(i), static_cast<Real>(j), static_cast<Real>(k)});

                Real dr = m_dx_[0];
                Real dl0 = m_dx_[1] * x[0];
                Real dl1 = m_dx_[1] * (x[0] + m_dx_[0]);
                Real dz = m_dx_[2];

                m_volume_[0](i, j, k) = 1.0;
                m_volume_[1](i, j, k) = dr;
                m_volume_[2](i, j, k) = dl0;
                m_volume_[3](i, j, k) = 0.5 * dr * (dl0 + dl1);
                m_volume_[4](i, j, k) = dz;
                m_volume_[5](i, j, k) = dr * dz;
                m_volume_[6](i, j, k) = dl0 * dz;
                m_volume_[7](i, j, k) = 0.5 * dr * (dl0 + dl1) * dz;
                m_volume_[8](i, j, k) = 1.0;

                m_inv_volume_[0](i, j, k) = 1.0 / m_volume_[0](i, j, k);
                m_inv_volume_[1](i, j, k) = 1.0 / m_volume_[1](i, j, k);
                m_inv_volume_[2](i, j, k) = 1.0 / m_volume_[2](i, j, k);
                m_inv_volume_[3](i, j, k) = 1.0 / m_volume_[3](i, j, k);
                m_inv_volume_[4](i, j, k) = 1.0 / m_volume_[4](i, j, k);
                m_inv_volume_[5](i, j, k) = 1.0 / m_volume_[5](i, j, k);
                m_inv_volume_[6](i, j, k) = 1.0 / m_volume_[6](i, j, k);
                m_inv_volume_[7](i, j, k) = 1.0 / m_volume_[7](i, j, k);
                m_inv_volume_[8](i, j, k) = 1.0 / m_volume_[8](i, j, k);

                dr = m_dx_[0];
                dl0 = m_dx_[1] * (x[0] - 0.5 * m_dx_[0]);
                dl1 = m_dx_[1] * (x[0] + 0.5 * m_dx_[0]);
                dz = m_dx_[2];

                m_dual_volume_[7](i, j, k) = 1.0;
                m_dual_volume_[6](i, j, k) = dr;
                m_dual_volume_[5](i, j, k) = dl0;
                m_dual_volume_[4](i, j, k) = 0.5 * dr * (dl0 + dl1);
                m_dual_volume_[3](i, j, k) = dz;
                m_dual_volume_[2](i, j, k) = dr * dz;
                m_dual_volume_[1](i, j, k) = dl0 * dz;
                m_dual_volume_[0](i, j, k) = 0.5 * dr * (dl0 + dl1) * dz;
                m_dual_volume_[8](i, j, k) = 1.0;

                m_inv_dual_volume_[0](i, j, k) = 1.0 / m_dual_volume_[0](i, j, k);
                m_inv_dual_volume_[1](i, j, k) = 1.0 / m_dual_volume_[1](i, j, k);
                m_inv_dual_volume_[2](i, j, k) = 1.0 / m_dual_volume_[2](i, j, k);
                m_inv_dual_volume_[3](i, j, k) = 1.0 / m_dual_volume_[3](i, j, k);
                m_inv_dual_volume_[4](i, j, k) = 1.0 / m_dual_volume_[4](i, j, k);
                m_inv_dual_volume_[5](i, j, k) = 1.0 / m_dual_volume_[5](i, j, k);
                m_inv_dual_volume_[6](i, j, k) = 1.0 / m_dual_volume_[6](i, j, k);
                m_inv_dual_volume_[7](i, j, k) = 1.0 / m_dual_volume_[7](i, j, k);
                m_inv_dual_volume_[8](i, j, k) = 1.0 / m_dual_volume_[8](i, j, k);
            }
}

}  // namespace mesh{
}  // namespace simpla{
