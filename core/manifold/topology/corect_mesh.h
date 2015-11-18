//! @file corectmesh.h

// Created by salmon on 15-7-2.
//

#ifndef SIMPLA_CORECTMESH_H
#define SIMPLA_CORECTMESH_H

#include <vector>

#include "../../gtl/macro.h"
#include "../../gtl/primitives.h"
#include "../../gtl/ntuple.h"
#include "../../gtl/type_traits.h"
#include "../../gtl/utilities/utilities.h"
#include "mesh_block.h"
#include "map_linear.h"

namespace simpla
{
namespace topology
{


struct CoRectMesh : public MeshBlock<>, private LinearMap
{
    static constexpr int ndims = 3;
    enum
    {
        DEFAULT_GHOST_WIDTH = 2
    };
private:

    typedef CoRectMesh this_type;
    typedef LinearMap map_type;
    typedef MeshBlock base_type;
public:
    using base_type::id_type;
    using base_type::id_tuple;
    using base_type::index_type;
    typedef id_type value_type;
    typedef size_t difference_type;
    typedef nTuple<Real, ndims> point_type;
    typedef nTuple<Real, ndims> vector_type;
    using base_type::index_tuple;


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




    bool m_is_valid_ = false;
public:

    CoRectMesh() : base_type()
    {

    }


    CoRectMesh(this_type const &other) :
            base_type(other), m_coords_min_(other.m_coords_min_), m_coords_max_(other.m_coords_max_), m_dx_(other.m_dx_)
    {
    }

    virtual  ~CoRectMesh() { }

    virtual void swap(this_type &other)
    {

        base_type::swap(other);
        map_type::swap(other);

        std::swap(m_coords_min_, other.m_coords_min_);
        std::swap(m_coords_max_, other.m_coords_max_);
        std::swap(m_dx_, other.m_dx_);


        deploy();
        other.deploy();

    }

    bool is_valid() const { return m_is_valid_; }

    template<typename TDict>
    void load(TDict const &dict)
    {
        box(dict["BaseManifold"]["Box"].template as<std::tuple<point_type, point_type> >());

        base_type::dimensions(
                dict["BaseManifold"]["Topology"]["Dimensions"].template as<index_tuple>(index_tuple{10, 1, 1}));
    }

    template<typename OS>
    OS &print(OS &os) const
    {

        os << "\t\tTopology = {" << std::endl
        << "\t\t Type = " << traits::type_id<this_type>::name() << "\",  }," << std::endl
        << "\t\t Extents = {" << box() << "}," << std::endl
        << "\t\t Dimensions = " << base_type::dimensions() << "," << std::endl
        << "\t\t}, " << std::endl;

        return os;
    }

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
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

    template<typename T0>
    void box(T0 const &b)
    {
        box(simpla::traits::get<0>(b), simpla::traits::get<1>(b));
    }

    std::tuple<point_type, point_type> box() const
    {
        return (std::make_tuple(m_coords_min_, m_coords_max_));
    }

    vector_type const &dx() const
    {
        return m_dx_;
    }

/**
 * @name  Coordinate map
 * @{
 *
 *        Topology Mesh       BaseManifold Mesh
 *                        map
 *              M      ---------->      G
 *              x                       y
 **/
private:

    using map_type::map;
    using map_type::inv_map;


public:

    virtual point_type point(id_type const &s) const { return std::move(map(base_type::coordinates(s))); }


    virtual point_type coordinates_local_to_global(std::tuple<id_type, point_type> const &t) const
    {
        return std::move(map(base_type::coordinates_local_to_global(t)));
    }

    virtual std::tuple<id_type, point_type> coordinates_global_to_local(point_type x, int n_id = 0) const
    {
        return std::move(base_type::coordinates_global_to_local(inv_map(x), n_id));
    }

    virtual id_type id(point_type x, int n_id = 0) const
    {
        return std::get<0>(base_type::coordinates_global_to_local(inv_map(x), n_id));
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

        base_type::deploy();

        auto dims = base_type::dimensions();

        map_type::set(base_type::box(), box(), dims);


        for (int i = 0; i < ndims; ++i)
        {
            ASSERT(dims[i] >= 1);

            ASSERT((m_coords_max_[i] - m_coords_min_[i] > EPSILON));

            m_dx_[i] = (m_coords_max_[i] - m_coords_min_[i]) / static_cast<Real>(dims[i]);
        }
    }

    template<typename TGeo>
    void update_volume(TGeo const &geo)
    {

        m_is_valid_ = true;

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



        base_type::get_element_volume_in_cell(geo, 0, m_volume_, m_inv_volume_, m_dual_volume_, m_inv_dual_volume_);

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


}; // struct CoRectMesh
}// namespace topology
} // namespace simpla

#endif //SIMPLA_CORECTMESH_H
