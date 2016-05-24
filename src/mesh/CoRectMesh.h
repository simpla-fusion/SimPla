/**
 *
 * @file corectmesh.h
 * Created by salmon on 15-7-2.
 *
 */

#ifndef SIMPLA_CORECTMESH_H
#define SIMPLA_CORECTMESH_H

#include <vector>

#include "../gtl/macro.h"
#include "../gtl/primitives.h"
#include "../gtl/nTuple.h"
#include "../gtl/type_traits.h"
#include "../../gtl/utilities.h"
#include "../../geometry/GeoAlgorithm.h"
#include "obsolete/MeshBlock.h"


#include "Mesh.h"

namespace simpla { namespace mesh
{
namespace tags { struct CoRectLinear; }

template<typename ...> class Mesh;

template<typename TMetric> using CoRectMesh=Mesh<tags::CoRectLinear>;


/**
 * @ingroup mesh
 *
 * @brief Uniform structured mesh
 */
template<>
struct Mesh<tags::CoRectLinear> : public MeshBlock
{
private:
    typedef Mesh<tags::CoRectLinear> this_type;
    typedef block_type base_type;
public:


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

    Mesh()  { }


    Mesh(this_type const &other) = delete;

    virtual  ~Mesh() { }


    virtual std::ostream &print(std::ostream &os, int indent = 1) const
    {

        os
        << std::setw(indent) << "\tGeometry={" << std::endl
        << std::setw(indent) << "\t\t Topology = { Type = \"RectMesh\",  }," << std::endl
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
//private:
//
//
//    point_type m_map_orig_ = {0, 0, 0};
//
//    point_type m_map_scale_ = {1, 1, 1};
//
//    point_type m_inv_map_orig_ = {0, 0, 0};
//
//    point_type m_inv_map_scale_ = {1, 1, 1};
//
//
//    point_type inv_map(point_type const &x) const
//    {
//
//        point_type res;
//
//        res[0] = std::fma(x[0], m_inv_map_scale_[0], m_inv_map_orig_[0]);
//
//        res[1] = std::fma(x[1], m_inv_map_scale_[1], m_inv_map_orig_[1]);
//
//        res[2] = std::fma(x[2], m_inv_map_scale_[2], m_inv_map_orig_[2]);
//
//        return std::move(res);
//    }
//
//    point_type map(point_type const &y) const
//    {
//
//        point_type res;
//
//
//        res[0] = std::fma(y[0], m_map_scale_[0], m_map_orig_[0]);
//
//        res[1] = std::fma(y[1], m_map_scale_[1], m_map_orig_[1]);
//
//        res[2] = std::fma(y[2], m_map_scale_[2], m_map_orig_[2]);
//
//        return std::move(res);
//    }
//
//public:
//
//    virtual point_type point(id_type const &s) const { return std::move(map(block_type::point(s))); }
//
//    virtual point_type coordinates_local_to_global(id_type s, point_type const &x) const
//    {
//        return std::move(map(block_type::coordinates_local_to_global(s, x)));
//    }
//
//    virtual point_type coordinates_local_to_global(std::tuple<id_type, point_type> const &t) const
//    {
//        return std::move(map(block_type::coordinates_local_to_global(t)));
//    }
//
//    virtual std::tuple<id_type, point_type> coordinates_global_to_local(point_type const &x, int n_id = 0) const
//    {
//        return std::move(block_type::coordinates_global_to_local(inv_map(x), n_id));
//    }
//
//    virtual id_type id(point_type const &x, int n_id = 0) const
//    {
//        return std::get<0>(block_type::coordinates_global_to_local(inv_map(x), n_id));
//    }

    using MeshBlock::point;
    using MeshBlock::coordinates_local_to_global;
    using MeshBlock::coordinates_global_to_local;
    using MeshBlock::id;

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

//        {
//
//
//            point_type src_min_, src_max_;
//
//            point_type dest_min, dest_max;
//
//            std::tie(src_min_, src_max_) = block_type::index_box();
//
//            std::tie(dest_min, dest_max) = box();
//
//
//            for (int i = 0; i < 3; ++i)
//            {
//                m_map_scale_[i] = (dest_max[i] - dest_min[i]) / (src_max_[i] - src_min_[i]);
//
//                m_inv_map_scale_[i] = (src_max_[i] - src_min_[i]) / (dest_max[i] - dest_min[i]);
//
//
//                m_map_orig_[i] = dest_min[i] - src_min_[i] * m_map_scale_[i];
//
//                m_inv_map_orig_[i] = src_min_[i] - dest_min[i] * m_inv_map_scale_[i];
//
//            }
//        }


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
    }


    template<typename T0, typename T1, typename ...Others>
    static constexpr auto inner_product(T0 const &v0, T1 const &v1, Others &&... others)
    DECL_RET_TYPE((v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2]))

}; // struct mesh
}// namespace mesh
} // namespace simpla

#endif //SIMPLA_CORECTMESH_H
