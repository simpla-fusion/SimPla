//
// Created by salmon on 16-10-9.
//

#ifndef SIMPLA_CYLINDRICALRECTMESH_H
#define SIMPLA_CYLINDRICALRECTMESH_H


#include <vector>
#include <iomanip>
#include <simpla/SIMPLA_config.h>

#include <simpla/toolbox/macro.h>
#include <simpla/toolbox/nTuple.h>
#include <simpla/toolbox/nTupleExt.h>
#include <simpla/toolbox/PrettyStream.h>
#include <simpla/toolbox/type_traits.h>
#include <simpla/toolbox/type_cast.h>
#include <simpla/toolbox/Log.h>

#include "MeshBlock.h"

namespace simpla { namespace mesh
{


/**
 * @ingroup mesh
 *
 * @brief Uniform structured get_mesh
 */
struct CylindricalRectMesh : public MeshBlock
{
private:
    typedef CylindricalRectMesh this_type;
    typedef MeshBlock base_type;
public:
    virtual bool is_a(std::type_info const &info) const { return typeid(this_type) == info; }

    virtual std::string get_class_name() const { return class_name(); }

    static std::string class_name() { return std::string("CylindricalRectMesh"); }


public:
    static constexpr int ndims = 3;

    CylindricalRectMesh() {}

    CylindricalRectMesh(this_type const &other) : MeshBlock(other) {};

    virtual  ~CylindricalRectMesh() {}

    std::vector<Real> m_volume_[9];
    std::vector<Real> m_inv_volume_[9];
    std::vector<Real> m_dual_volume_[9];
    std::vector<Real> m_inv_dual_volume_[9];

public:


    virtual Real volume(MeshEntityId s) const { return m_volume_[m::node_id(s)][m::sub_index(s)]; }

    virtual Real dual_volume(MeshEntityId s) const { return m_dual_volume_[m::node_id(s)][m::sub_index(s)]; }

    virtual Real inv_volume(MeshEntityId s) const { return m_inv_volume_[m::node_id(s)][m::sub_index(s)]; }

    virtual Real inv_dual_volume(MeshEntityId s) const { return m_inv_dual_volume_[m::node_id(s)][m::sub_index(s)]; }

    void deploy()
    {

        base_type::deploy();
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
        for (int i = 0; i < ndims; ++i)
        {



//            m_dx_[i] = (m_dims_[i] <= 1) ? 1 : (m_coords_upper_[i] - m_coords_lower_[i]) /
//                                               static_cast<Real>( m_dims_[i]);
//
//            m_inv_dx_[i] = (m_dims_[i] <= 1) ? 0 : static_cast<Real>(1.0) / m_dx_[i];
//
//            m_l2g_scale_[i] = (m_dims_[i] <= 1) ? 0 : m_dx_[i];
//            m_l2g_shift_ = m_coords_lower_;
//
//            m_g2l_scale_[i] = (m_dims_[i] <= 1) ? 0 : static_cast<Real>(1.0) / m_dx_[i];
//            m_g2l_shift_[i] = (m_dims_[i] <= 1) ? 0 : -m_coords_lower_[i] * m_g2l_scale_[i];

        }


//        m_volume_[0 /*000*/] = 1;
//        m_volume_[1 /*001*/] = m_dx_[0];
//        m_volume_[2 /*010*/] = m_dx_[1];
//        m_volume_[4 /*100*/] = m_dx_[2];
//        m_volume_[3 /*011*/] = m_dx_[0] * m_dx_[1];
//        m_volume_[5 /*101*/] = m_dx_[2] * m_dx_[0];
//        m_volume_[6 /*110*/] = m_dx_[1] * m_dx_[2];
//        m_volume_[7 /*110*/] = m_dx_[0] * m_dx_[1] * m_dx_[2];
//
//
//        m_dual_volume_[0 /*000*/] = m_volume_[7];
//        m_dual_volume_[1 /*001*/] = m_volume_[6];
//        m_dual_volume_[2 /*010*/] = m_volume_[5];
//        m_dual_volume_[4 /*100*/] = m_volume_[3];
//        m_dual_volume_[3 /*011*/] = m_volume_[4];
//        m_dual_volume_[5 /*101*/] = m_volume_[2];
//        m_dual_volume_[6 /*110*/] = m_volume_[1];
//        m_dual_volume_[7 /*110*/] = m_volume_[0];
//
//
//        m_inv_dx_[0] = (m_dims_[0] == 1) ? 1 : 1 / m_dx_[0];
//        m_inv_dx_[1] = (m_dims_[1] == 1) ? 1 : 1 / m_dx_[1];
//        m_inv_dx_[2] = (m_dims_[2] == 1) ? 1 : 1 / m_dx_[2];
//
//
//        m_inv_volume_[0 /*000*/] = 1;
//        m_inv_volume_[1 /*001*/] = m_inv_dx_[0];
//        m_inv_volume_[2 /*010*/] = m_inv_dx_[1];
//        m_inv_volume_[4 /*100*/] = m_inv_dx_[2];
//        m_inv_volume_[3 /*011*/] = m_inv_dx_[0] * m_inv_dx_[1];
//        m_inv_volume_[5 /*101*/] = m_inv_dx_[2] * m_inv_dx_[0];
//        m_inv_volume_[6 /*110*/] = m_inv_dx_[1] * m_inv_dx_[2];
//        m_inv_volume_[7 /*110*/] = m_inv_dx_[0] * m_inv_dx_[1] * m_inv_dx_[2];
//
//
//        m_inv_dual_volume_[0 /*000*/] = m_inv_volume_[7];
//        m_inv_dual_volume_[1 /*001*/] = m_inv_volume_[6];
//        m_inv_dual_volume_[2 /*010*/] = m_inv_volume_[5];
//        m_inv_dual_volume_[4 /*100*/] = m_inv_volume_[3];
//        m_inv_dual_volume_[3 /*011*/] = m_inv_volume_[4];
//        m_inv_dual_volume_[5 /*101*/] = m_inv_volume_[2];
//        m_inv_dual_volume_[6 /*110*/] = m_inv_volume_[1];
//        m_inv_dual_volume_[7 /*110*/] = m_inv_volume_[0];
//
//
//        m_inv_dx_[0] = (m_dims_[0] <= 1) ? 0 : m_inv_dx_[0];
//        m_inv_dx_[1] = (m_dims_[1] <= 1) ? 0 : m_inv_dx_[1];
//        m_inv_dx_[2] = (m_dims_[2] <= 1) ? 0 : m_inv_dx_[2];
//
//
//        m_inv_volume_[1 /*001*/] = (m_dims_[0] <= 1) ? 0 : m_inv_dx_[0];
//        m_inv_volume_[2 /*010*/] = (m_dims_[1] <= 1) ? 0 : m_inv_dx_[1];
//        m_inv_volume_[4 /*100*/] = (m_dims_[2] <= 1) ? 0 : m_inv_dx_[2];
//
//

    }


}; // struct  Mesh
}} // namespace simpla // namespace get_mesh


#endif //SIMPLA_CYLINDRICALRECTMESH_H
