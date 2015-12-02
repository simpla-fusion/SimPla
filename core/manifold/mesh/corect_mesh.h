/**
 *
 * @file corectmesh.h
 * Created by salmon on 15-7-2.
 *
 */

#ifndef SIMPLA_CORECTMESH_H
#define SIMPLA_CORECTMESH_H

#include <vector>

#include "../../gtl/macro.h"
#include "../../gtl/primitives.h"
#include "../../gtl/ntuple.h"
#include "../../gtl/type_traits.h"
#include "../../gtl/utilities/utilities.h"
#include "../../geometry/geo_algorithm.h"
#include "mesh.h"
#include "mesh_block.h"
#include "map_linear.h"

namespace simpla { namespace mesh
{
namespace tags
{
struct corect_linear;
}
template<typename TMetric>
using CoRectMesh=Mesh<TMetric, tags::corect_linear>;

/**
 * @ingroup mesh
 *
 * @brief Uniform structured mesh
 */
template<typename TMetric>
struct Mesh<TMetric, tags::corect_linear> : public TMetric, public MeshBlock, private LinearMap
{
    static constexpr int ndims = 3;
    enum
    {
        DEFAULT_GHOST_WIDTH = 2
    };
private:

    typedef Mesh<TMetric, tags::corect_linear> this_type;
    typedef LinearMap map_type;
    typedef MeshBlock base_type;
public:
    using base_type::id_type;
    using base_type::id_tuple;
    using base_type::index_type;
    using base_type::index_tuple;
    using base_type::range_type;

    typedef id_type value_type;
    typedef size_t difference_type;
    typedef nTuple <Real, ndims> point_type;
    typedef nTuple <Real, ndims> vector_type;


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

    Mesh() : base_type()
    {

    }


    Mesh(this_type const &other) :
            base_type(other), m_coords_min_(other.m_coords_min_),
            m_coords_max_(other.m_coords_max_),
            m_dx_(other.m_dx_)
    {
    }

    virtual  ~Mesh() { }

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

    static std::string topology_type() { return "CoRectMesh"; }

    bool is_valid() const { return m_is_valid_; }

    template<typename TDict>
    void load(TDict const &dict)
    {
        try
        {
            box(dict["Geometry"]["Box"].template as<std::tuple<point_type, point_type> >());

            base_type::dimensions(
                    dict["Geometry"]["Topology"]["Dimensions"].template as<index_tuple>(index_tuple{10, 1, 1}));

        }
        catch (std::runtime_error const &e)
        {
            SHOW_ERROR << e.what() << std::endl;

            THROW_EXCEPTION_PARSER_ERROR("Geometry is not correctly loaded!");

        }
    }

    template<typename OS>
    OS &print(OS &os) const
    {

        os
        << "\tGeometry={" << std::endl
        << "\t\t Topology = { Type = \"CoRectMesh\",  }," << std::endl
        << "\t\t Box = {" << box() << "}," << std::endl
        << "\t\t Dimensions = " << base_type::dimensions() << "," << std::endl
        << "\t\t}, " << std::endl
        << "\t}, " << std::endl;

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

    std::tuple<point_type, point_type> box(id_type const &s) const
    {
        return std::make_tuple(point(s - base_type::_DA), point(s + base_type::_DA));
    };


    vector_type const &dx() const
    {
        return m_dx_;
    }

    int get_vertices(int node_id, id_type s, point_type *p = nullptr) const
    {

        int num = base_type::get_adjacent_cells(VERTEX, node_id, s);

        if (p != nullptr)
        {
            id_type neighbour[num];

            base_type::get_adjacent_cells(VERTEX, node_id, s, neighbour);

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
 *        Topology Mesh       Geometry Mesh
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

    virtual id_type id(point_type const &x, int n_id = 0) const
    {
        return std::get<0>(base_type::coordinates_global_to_local(inv_map(x), n_id));
    }

    std::tuple<index_tuple, index_tuple> index_box(std::tuple<point_type, point_type> const &b) const
    {

        point_type b0, b1, x0, x1;

        std::tie(b0, b1) = local_box();
        std::tie(x0, x1) = b;

        if (geometry::box_intersection(b0, b1, &x0, &x1))
        {
            return std::make_tuple(base_type::unpack_index(id(x0)),
                                   base_type::unpack_index(id(x1) + (base_type::_DA << 1)));

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

        base_type::deploy();

        auto dims = base_type::dimensions();

        map_type::set(base_type::box(), box(), dims);


        for (int i = 0; i < ndims; ++i)
        {
            ASSERT(dims[i] >= 1);

            ASSERT((m_coords_max_[i] - m_coords_min_[i] > EPSILON));

            m_dx_[i] = (m_coords_max_[i] - m_coords_min_[i]) / static_cast<Real>(dims[i]);
        }


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



        base_type::get_element_volume_in_cell(*this, 0, m_volume_, m_inv_volume_,
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

}; // struct Mesh
}// namespace mesh
} // namespace simpla

#endif //SIMPLA_CORECTMESH_H
