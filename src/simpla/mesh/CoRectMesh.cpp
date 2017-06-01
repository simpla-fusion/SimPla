//
// Created by salmon on 17-6-1.
//

#include "CoRectMesh.h"

#include <simpla/utilities/EntityIdCoder.h>
namespace simpla {
namespace mesh {
REGISTER_CREATOR(CoRectMesh);

inline void CoRectMesh::InitializeData(Real time_now) {
    StructuredMesh::InitializeData(time_now);
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
    //    m_scale_ = GetChart()->GetDx();
    size_tuple m_dims_ = GetBlock()->GetDimensions();

    //    m_volume_[0 /*000*/] = 1;
    //    m_volume_[1 /*001*/] = (m_dims_[0] == 1) ? 1 : m_scale_[0];
    //    m_volume_[2 /*010*/] = (m_dims_[1] == 1) ? 1 : m_scale_[1];
    //    m_volume_[4 /*100*/] = (m_dims_[2] == 1) ? 1 : m_scale_[2];
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
}  // namespace mesh {
}  // namespace simpla {