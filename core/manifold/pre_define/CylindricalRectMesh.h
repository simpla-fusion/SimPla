/**
 * @file CylindricalRectMesh.h
 * @author salmon
 * @date 2016-01-11.
 */

#ifndef SIMPLA_CYLINDRICALRECTMESH_H
#define SIMPLA_CYLINDRICALRECTMESH_H

#include <limits>
#include "../../geometry/csCylindrical.h"
#include "../mesh/MeshBlock.h"

#include "../../gtl/design_pattern/singleton_holder.h"
#include "../../gtl/utilities/memory_pool.h"

namespace simpla { namespace mesh
{


class CylindricalRectMesh : public geometry::CylindricalMetric, public MeshBlock
{

private:
    typedef CylindricalRectMesh this_type;
public:

    SP_OBJECT_HEAD(CylindricalRectMesh, MeshBlock)

    typedef geometry::CylindricalMetric metric_type;

    typedef typename metric_type::cs coordinate_system_type;

    typedef MeshBlock block_type;

    HAS_PROPERTIES;

    using block_type::ndims;
    using block_type::id_type;
    using block_type::id_tuple;
    using block_type::index_type;
    using block_type::index_tuple;
    using block_type::range_type;
    using block_type::difference_type;


    typedef typename metric_type::point_type point_type;
    typedef typename metric_type::vector_type vector_type;
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


    point_type m_coords_min_ = {0, 0, 0};

    point_type m_coords_max_ = {1, 1, 1};

    vector_type m_dx_ = {1, 1, 1};; //!< width of cell, except m_dx_[i]=0 when m_dims_[i]==1

public:

    CylindricalRectMesh() : block_type() { }


    CylindricalRectMesh(this_type const &other) = delete;

    virtual  ~CylindricalRectMesh() { }


    virtual std::ostream &print(std::ostream &os, int indent = 1) const
    {

        os
        << std::setw(indent) << "\tGeometry={" << std::endl
        << std::setw(indent) << "\t\t Topology = { Type = \"CylindricalRectMesh\",  }," << std::endl
        << std::setw(indent) << "\t\t Box = {" << box() << "}," << std::endl
        << std::setw(indent) << "\t\t Dimensions = " << block_type::dimensions() << "," << std::endl
        << std::setw(indent) << "\t\t}, " << std::endl
        << std::setw(indent) << "\t}" << std::endl;

        return os;
    }




//================================================================================================
// @name Coordinates dependent

//    point_type epsilon() const { return EPSILON * m_from_topology_scale_; }

    template<typename X0, typename X1>
    void box(X0 const &x0, X1 const &x1)
    {
        m_coords_min_ = x0;
        m_coords_max_ = x1;
    }


    void box(box_type const &b) { std::tie(m_coords_min_, m_coords_max_) = b; }

    box_type box() const { return (std::make_tuple(m_coords_min_, m_coords_max_)); }

    box_type box(id_type const &s) const
    {
        return std::make_tuple(point(s - block_type::_DA), point(s + block_type::_DA));
    }


    vector_type const &dx() const { return m_dx_; }

    int get_vertices(int node_id, id_type s, point_type *p = nullptr) const
    {

        int num = block_type::get_adjacent_cells(VERTEX, node_id, s);

        if (p != nullptr)
        {
            id_type neighbour[num];

            block_type::get_adjacent_cells(VERTEX, node_id, s, neighbour);

            for (int i = 0; i < num; ++i)
            {
                p[i] = point(neighbour[i]);
            }

        }


        return num;
    }

/**
 * @name  Coordinate map
 * @{
 *
 *        Topology mesh       geometry mesh
 *                        map
 *              M      ---------->      G
 *              x                       y
 **/
private:


    point_type m_map_orig_ = {0, 0, 0};

    point_type m_map_scale_ = {1, 1, 1};

    point_type m_inv_map_orig_ = {0, 0, 0};

    point_type m_inv_map_scale_ = {1, 1, 1};


    point_type inv_map(point_type const &x) const
    {

        point_type res;

        res[0] = std::fma(x[0], m_inv_map_scale_[0], m_inv_map_orig_[0]);

        res[1] = std::fma(x[1], m_inv_map_scale_[1], m_inv_map_orig_[1]);

        res[2] = std::fma(x[2], m_inv_map_scale_[2], m_inv_map_orig_[2]);

        return std::move(res);
    }

    point_type map(point_type const &y) const
    {

        point_type res;


        res[0] = std::fma(y[0], m_map_scale_[0], m_map_orig_[0]);

        res[1] = std::fma(y[1], m_map_scale_[1], m_map_orig_[1]);

        res[2] = std::fma(y[2], m_map_scale_[2], m_map_orig_[2]);

        return std::move(res);
    }

public:

    virtual point_type point(id_type const &s) const { return std::move(map(block_type::point(s))); }


    virtual point_type coordinates_local_to_global(std::tuple<id_type, point_type> const &t) const
    {
        return std::move(map(block_type::coordinates_local_to_global(t)));
    }

    virtual std::tuple<id_type, point_type> coordinates_global_to_local(point_type const &x, int n_id = 0) const
    {
        return std::move(block_type::coordinates_global_to_local(inv_map(x), n_id));
    }

    virtual id_type id(point_type const &x, int n_id = 0) const
    {
        return std::get<0>(block_type::coordinates_global_to_local(inv_map(x), n_id));
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

        {


            point_type src_min_, src_max_;

            point_type dest_min, dest_max;

            std::tie(src_min_, src_max_) = block_type::index_box();

            std::tie(dest_min, dest_max) = box();


            for (int i = 0; i < 3; ++i)
            {
                m_map_scale_[i] = (dest_max[i] - dest_min[i]) / (src_max_[i] - src_min_[i]);

                m_inv_map_scale_[i] = (src_max_[i] - src_min_[i]) / (dest_max[i] - dest_min[i]);


                m_map_orig_[i] = dest_min[i] - src_min_[i] * m_map_scale_[i];

                m_inv_map_orig_[i] = src_min_[i] - dest_min[i] * m_inv_map_scale_[i];

            }
        }


        for (int i = 0; i < ndims; ++i)
        {
            ASSERT(dims[i] >= 1);

            ASSERT((m_coords_max_[i] - m_coords_min_[i] > EPSILON));

            m_dx_[i] = (m_coords_max_[i] - m_coords_min_[i]) / static_cast<Real>(dims[i]);
        }



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

//        auto dims = geo.dimensions();
//
//
//#define NOT_ZERO(_V_) ((_V_<EPSILON)?1.0:(_V_))
//        m_volume_[0] = 1.0;
//
//        m_volume_[1/* 001*/] = (dims[0] > 1) ? m_dx_[0] : 1.0;
//        m_volume_[2/* 010*/] = (dims[1] > 1) ? m_dx_[1] : 1.0;
//        m_volume_[4/* 100*/] = (dims[2] > 1) ? m_dx_[2] : 1.0;
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

}}  // namespace mesh // namespace simpla
#endif //SIMPLA_CYLINDRICALRECTMESH_H
