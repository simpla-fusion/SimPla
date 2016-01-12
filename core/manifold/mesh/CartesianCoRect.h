/**
 * @file CartesianCoRect.h
 * @author salmon
 * @date 2016-01-12.
 */

#ifndef SIMPLA_CARTESIANCORECT_H
#define SIMPLA_CARTESIANCORECT_H

#include <vector>

#include "../../gtl/macro.h"
#include "../../gtl/primitives.h"
#include "../../gtl/ntuple.h"
#include "../../gtl/type_traits.h"
#include "../../gtl/utilities/utilities.h"
#include "../../geometry/GeoAlgorithm.h"
#include "../../geometry/csCartesian.h"

#include "MeshBlock.h"

namespace simpla { namespace mesh
{


/**
 * @ingroup mesh
 *
 * @brief Uniform structured mesh
 */
struct CartesianCoRect : public geometry::CartesianMetric, public MeshBlock
{

private:
    typedef geometry::CartesianMetric metric_type;

    typedef CartesianCoRect this_type;

    typedef MeshBlock block_type;

public:
    SP_OBJECT_HEAD(CartesianCoRect, MeshBlock)


    using block_type::ndims;
    using block_type::id_type;
    using block_type::id_tuple;
    using block_type::index_type;
    using block_type::index_tuple;
    using block_type::range_type;
    using block_type::difference_type;


    using typename metric_type::scalar_type;
    using typename metric_type::point_type;
    using typename metric_type::vector_type;
    typedef std::tuple<point_type, point_type> box_type;


    /**
 *
 *   -----------------------------5
 *   |                            |
 *   |     ---------------4       |
 *   |     |              |       |
 *   |     |  ********3   |       |
 *   |     |  *       *   |       |
 *   |     |  *       *   |       |
 *   |     |  *       *   |       |
 *   |     |  2********   |       |
 *   |     1---------------       |
 *   0-----------------------------
 *
 *	5-0 = dimensions
 *	4-1 = e-d = ghosts
 *	2-1 = counts
 *
 *	0 = id_begin
 *	5 = id_end
 *
 *	1 = id_local_outer_begin
 *	4 = id_local_outer_end
 *
 *	2 = id_local_inner_begin
 *	3 = id_local_inner_end
 *
 *
 */


public:

    CartesianCoRect() : block_type() { }

    virtual  ~CartesianCoRect() { }

    CartesianCoRect(this_type const &other) = delete;


    virtual std::ostream &print(std::ostream &os, int indent = 1) const
    {

        os
        << std::setw(indent) << "\tGeometry={" << std::endl
        << std::setw(indent) << "\t\t Topology = { Type = \"CartesianCoRect\",  }," << std::endl
        << std::setw(indent) << "\t\t Box = {" << box() << "}," << std::endl
        << std::setw(indent) << "\t\t Dimensions = " << block_type::dimensions() << "," << std::endl
        << std::setw(indent) << "\t\t}, " << std::endl
        << std::setw(indent) << "\t}" << std::endl;

        return os;
    }


    std::tuple<index_tuple, index_tuple> index_box(std::tuple<point_type, point_type> const &b) const
    {

        point_type b0, b1, x0, x1;

        std::tie(b0, b1) = local_index_box();

        std::tie(x0, x1) = b;

        if (geometry::box_intersection(b0, b1, &x0, &x1))
        {
            return std::make_tuple(block_type::unpack_index(id(x0)),
                                   block_type::unpack_index(id(x1) + (block_type::_DA << 1)));

        }
        else
        {
            index_tuple i0, i1;
            i0 = 0;
            i1 = 0;
            return std::make_tuple(i0, i1);
        }

    }


private:
    Real m_volume_[9];
    Real m_inv_volume_[9];
    Real m_dual_volume_[9];
    Real m_inv_dual_volume_[9];
public:


    virtual Real volume(id_type s) const { return m_volume_[node_id(s)]; }

    virtual Real dual_volume(id_type s) const { return m_dual_volume_[node_id(s)]; }

    virtual Real inv_volume(id_type s) const { return m_inv_volume_[node_id(s)]; }

    virtual Real inv_dual_volume(id_type s) const { return m_inv_dual_volume_[node_id(s)]; }


    virtual void deploy()
    {

        block_type::deploy();

        auto dims = block_type::dimensions();


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



        block_type::get_element_volume_in_cell(*this, 0, m_volume_, m_inv_volume_,
                                               m_dual_volume_, m_inv_dual_volume_);

//        auto m_dx_ = block_type::dx();
//
//
//#define NOT_ZERO(_V_) ((_V_<EPSILON)?1.0:(_V_))
//        m_volume_[0] = 1.0;
//
//        m_volume_[1/* 001*/] = m_dx_[0];
//        m_volume_[2/* 010*/] = m_dx_[1];
//        m_volume_[4/* 100*/] = m_dx_[2];
//
////    m_volume_[1/* 001*/] = (m_dx_[0] <= EPSILON) ? 1 : m_dx_[0];
////    m_volume_[2/* 010*/] = (m_dx_[1] <= EPSILON) ? 1 : m_dx_[1];
////    m_volume_[4/* 100*/] = (m_dx_[2] <= EPSILON) ? 1 : m_dx_[2];
//
//        m_volume_[3] /* 011 */= m_volume_[1] * m_volume_[2];
//        m_volume_[5] /* 101 */= m_volume_[4] * m_volume_[1];
//        m_volume_[6] /* 110 */= m_volume_[2] * m_volume_[4];
//        m_volume_[7] /* 111 */= m_volume_[1] * m_volume_[2] * m_volume_[4];
//
//        m_dual_volume_[7] = 1.0;
//
//        m_dual_volume_[6] = m_volume_[1];
//        m_dual_volume_[5] = m_volume_[2];
//        m_dual_volume_[3] = m_volume_[4];
//
////    m_dual_volume_[6] = (m_dx_[0] <= EPSILON) ? 1 : m_dx_[0];
////    m_dual_volume_[5] = (m_dx_[1] <= EPSILON) ? 1 : m_dx_[1];
////    m_dual_volume_[3] = (m_dx_[2] <= EPSILON) ? 1 : m_dx_[2];
//
//        m_dual_volume_[4] /* 011 */= m_dual_volume_[6] * m_dual_volume_[5];
//        m_dual_volume_[2] /* 101 */= m_dual_volume_[3] * m_dual_volume_[6];
//        m_dual_volume_[1] /* 110 */= m_dual_volume_[5] * m_dual_volume_[3];
//
//        m_dual_volume_[0] /* 111 */= m_dual_volume_[6] * m_dual_volume_[5] * m_dual_volume_[3];
//
//        m_inv_volume_[7] = 1.0;
//
//        m_inv_volume_[1/* 001 */] = (dims[0] > 1) ? 1.0 / m_volume_[1] : 0;
//        m_inv_volume_[2/* 010 */] = (dims[1] > 1) ? 1.0 / m_volume_[2] : 0;
//        m_inv_volume_[4/* 100 */] = (dims[2] > 1) ? 1.0 / m_volume_[4] : 0;
//
//        m_inv_volume_[3] /* 011 */= NOT_ZERO(m_inv_volume_[1]) * NOT_ZERO(m_inv_volume_[2]);
//        m_inv_volume_[5] /* 101 */= NOT_ZERO(m_inv_volume_[4]) * NOT_ZERO(m_inv_volume_[1]);
//        m_inv_volume_[6] /* 110 */= NOT_ZERO(m_inv_volume_[2]) * NOT_ZERO(m_inv_volume_[4]);
//        m_inv_volume_[7] /* 111 */=
//                NOT_ZERO(m_inv_volume_[1]) * NOT_ZERO(m_inv_volume_[2]) * NOT_ZERO(m_inv_volume_[4]);
//
//        m_inv_dual_volume_[7] = 1.0;
//
//        m_inv_dual_volume_[6/* 110 */] = (dims[0] > 1) ? 1.0 / m_dual_volume_[6] : 0;
//        m_inv_dual_volume_[5/* 101 */] = (dims[1] > 1) ? 1.0 / m_dual_volume_[5] : 0;
//        m_inv_dual_volume_[3/* 001 */] = (dims[2] > 1) ? 1.0 / m_dual_volume_[3] : 0;
//
//        m_inv_dual_volume_[4] /* 011 */= NOT_ZERO(m_inv_dual_volume_[6]) * NOT_ZERO(m_inv_dual_volume_[5]);
//        m_inv_dual_volume_[2] /* 101 */= NOT_ZERO(m_inv_dual_volume_[3]) * NOT_ZERO(m_inv_dual_volume_[6]);
//        m_inv_dual_volume_[1] /* 110 */= NOT_ZERO(m_inv_dual_volume_[5]) * NOT_ZERO(m_inv_dual_volume_[3]);
//        m_inv_dual_volume_[0] /* 111 */=
//                NOT_ZERO(m_inv_dual_volume_[6]) * NOT_ZERO(m_inv_dual_volume_[5]) * NOT_ZERO(m_inv_dual_volume_[3]);
//#undef NOT_ZERO

    }


    template<typename T0, typename T1, typename ...Others>
    static constexpr auto inner_product(T0 const &v0, T1 const &v1, Others &&... others)
    DECL_RET_TYPE((v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2]))

}; // struct mesh
}} // namespace mesh // namespace simpla
#endif //SIMPLA_CARTESIANCORECT_H
