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
 * @brief Uniform structured get_mesh
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


//    std::tuple<index_tuple, index_tuple> index_box(std::tuple<point_type, point_type> const &b) const
//    {
//
//        point_type b0, b1, x0, x1;
//
//        std::tie(b0, b1) = local_index_box();
//
//        std::tie(x0, x1) = b;
//
//        if (geometry::box_intersection(b0, b1, &x0, &x1))
//        {
//            return std::make_tuple(m::unpack_index(id(x0)),
//                                   m::unpack_index(id(x1) + (m::_DA << 1)));
//
//        }
//        else
//        {
//            index_tuple i0, i1;
//            i0 = 0;
//            i1 = 0;
//            return std::make_tuple(i0, i1);
//        }
//
//    }


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
//
//        m::get_volumes(m_volume_, m_inv_volume_,
//                                m_dual_volume_, m_inv_dual_volume_);
    }


    template<typename T0, typename T1, typename ...Others>
    static constexpr auto inner_product(T0 const &v0, T1 const &v1, Others &&... others)
    DECL_RET_TYPE((v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2]))

}; // struct get_mesh
}} // namespace get_mesh // namespace simpla
#endif //SIMPLA_CARTESIANCORECT_H
