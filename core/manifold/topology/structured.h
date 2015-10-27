//
// Created by salmon on 15-7-2.
//

#ifndef SIMPLA_TOPOLOGY_H
#define SIMPLA_TOPOLOGY_H

#include <vector>

#include "../../gtl/macro.h"
#include "../../gtl/primitives.h"
#include "../../gtl/ntuple.h"
#include "../../gtl/type_traits.h"
#include "mesh_ids.h"
#include "topology_common.h"

#include "../../gtl/utilities/utilities.h"


namespace simpla
{
template<typename...>
struct Topology;

namespace topology
{


struct StructuredMesh : public MeshIDs_<4>
{
    static constexpr int ndims = 3;
    enum
    {
        DEFAULT_GHOST_WIDTH = 2
    };
private:

    typedef StructuredMesh this_type;
    typedef MeshIDs_<4> m;

public:
    using m::id_type;
    using m::id_tuple;
    using m::index_type;
    typedef id_type value_type;
    typedef size_t difference_type;
    typedef nTuple<Real, ndims> point_type;
    typedef nTuple<Real, ndims> vector_type;
    using m::index_tuple;


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
    index_tuple m_min_;
    index_tuple m_max_;
    index_tuple m_local_min_;
    index_tuple m_local_max_;
    index_tuple m_memory_min_;
    index_tuple m_memory_max_;


    point_type m_coords_min_ = {0, 0, 0};

    point_type m_coords_max_ = {1, 1, 1};

    vector_type m_dx_ = {1, 1, 1};; //!< width of cell, except m_dx_[i]=0 when m_dims_[i]==1


    point_type m_from_topology_orig_ = {0, 0, 0};

    point_type m_to_topology_orig_ = {0, 0, 0};

    point_type m_to_topology_scale_ = {1, 1, 1};

    point_type m_from_topology_scale_ = {1, 1, 1};

public:

    StructuredMesh()
    {
        m_min_ = 0;
        m_max_ = 0;
        m_local_min_ = m_min_;
        m_local_max_ = m_max_;
        m_memory_min_ = m_min_;
        m_memory_max_ = m_max_;
    }


    StructuredMesh(StructuredMesh const &other) :

            m_min_(other.m_min_),

            m_max_(other.m_max_),

            m_local_min_(other.m_local_min_),

            m_local_max_(other.m_local_max_),

            m_memory_min_(other.m_memory_min_),

            m_memory_max_(other.m_memory_max_),

            m_coords_min_(other.m_coords_min_),

            m_coords_max_(other.m_coords_max_),

            m_dx_(other.m_dx_),

            m_from_topology_orig_(other.m_from_topology_orig_),

            m_to_topology_orig_(other.m_to_topology_orig_),

            m_to_topology_scale_(other.m_to_topology_scale_),

            m_from_topology_scale_(other.m_from_topology_scale_)
    {

    }

    virtual  ~StructuredMesh() { }

    virtual void swap(this_type &other)
    {
        std::swap(m_min_, other.m_min_);
        std::swap(m_max_, other.m_max_);
        std::swap(m_local_min_, other.m_local_min_);
        std::swap(m_local_max_, other.m_local_max_);
        std::swap(m_memory_min_, other.m_memory_min_);
        std::swap(m_memory_max_, other.m_memory_max_);


        std::swap(m_coords_min_, other.m_coords_min_);
        std::swap(m_coords_max_, other.m_coords_max_);
        std::swap(m_dx_, other.m_dx_);
        std::swap(m_from_topology_orig_, other.m_from_topology_orig_);
        std::swap(m_to_topology_orig_, other.m_to_topology_orig_);
        std::swap(m_to_topology_scale_, other.m_to_topology_scale_);
        std::swap(m_from_topology_scale_, other.m_from_topology_scale_);

        deploy();
        other.deploy();

    }

    template<typename TDict>
    void load(TDict const &dict)
    {
        box(dict["Geometry"]["Box"].template as<std::tuple<point_type, point_type> >());

        dimensions(dict["Topology"]["Dimensions"].template as<index_tuple>());
    }

    template<typename OS>
    OS &print(OS &os) const
    {

        os << "\t\tTopology = {" << std::endl
        << "\t\t Type = \"StructuredMesh\"," << std::endl
        << "\t\t Extents = {" << box() << "}," << std::endl
        << "\t\t Count = {}," << std::endl
        << "\t\t}, " << std::endl;

        return os;
    }


    virtual bool is_valid() const { return true; }


    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }


    template<typename TD>
    void dimensions(TD const &d)
    {
        m_max_ = d;
        m_min_ = 0;
    }

    index_tuple dimensions() const
    {
        index_tuple res;

        res = m_max_ - m_min_;

        return std::move(res);
    }

    template<typename T0, typename T1>
    void index_box(T0 const &min, T1 const &max)
    {
        m_min_ = min;
        m_max_ = max;
    };


    auto index_box() const
    DECL_RET_TYPE((std::make_tuple(m::point(m_min_), m::point(m_max_))))

    auto local_index_box() const
    DECL_RET_TYPE((std::make_tuple(m::point(m_local_min_), m::point(m_local_max_))))

    auto memory_index_box() const
    DECL_RET_TYPE((std::make_tuple(m::point(m_memory_min_), m::point(m_memory_max_))))


    template<typename T>
    bool in_box(T const &x) const
    {
        return (m_local_min_[1] <= x[1]) && (m_local_min_[2] <= x[2]) && (m_local_min_[0] <= x[0])  //
               && (m_local_max_[1] > x[1]) && (m_local_max_[2] > x[2]) && (m_local_max_[0] > x[0]);

    }

    bool in_box(id_type s) const
    {
        return in_box(m::unpack_index(s));
    }

    template<int I>
    range_type range() const { return m::template range<I>(m_local_min_, m_local_max_); }


    template<size_t IFORM>
    auto max_hash() const
    DECL_RET_TYPE((m::hash(m::pack_index(m_memory_max_ - 1, m::template sub_index_to_id<IFORM>(3UL)),
                           m_memory_min_, m_memory_max_)))


    size_t hash(id_type const &s) const { return static_cast<size_t>(m::hash(s, m_memory_min_, m_memory_max_)); }

    //================================================================================================
    // @name Coordinates dependent
private:

public:


    point_type epsilon() const { return EPSILON * m_from_topology_scale_; }


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

    constexpr std::tuple<point_type, point_type> box() const
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
 *        Topology Mesh       Geometry Mesh
 *                        map
 *              M      ---------->      G
 *              x                       y
 **/
private:

    point_type map(point_type const &x) const
    {

        point_type res;


        res[0] = std::fma(x[0], m_from_topology_scale_[0], m_from_topology_orig_[0]);

        res[1] = std::fma(x[1], m_from_topology_scale_[1], m_from_topology_orig_[1]);

        res[2] = std::fma(x[2], m_from_topology_scale_[2], m_from_topology_orig_[2]);


        return std::move(res);
    }

    point_type inv_map(point_type const &y) const
    {

        point_type res;


        res[0] = std::fma(y[0], m_to_topology_scale_[0], m_to_topology_orig_[0]);

        res[1] = std::fma(y[1], m_to_topology_scale_[1], m_to_topology_orig_[1]);

        res[2] = std::fma(y[2], m_to_topology_scale_[2], m_to_topology_orig_[2]);

        return std::move(res);
    }


public:

    point_type point(id_type const &s) const { return std::move(map(m::coordinates(s))); }

/**
 * @bug: truncation error of coordinates transform larger than 1000
 *     epsilon (1e4 epsilon for cylindrical coordinates)
 * @param args
 * @return
 *
 */
    point_type coordinates_local_to_global(std::tuple<id_type, point_type> const &t) const
    {
        return std::move(map(m::coordinates_local_to_global(t)));
    }

    std::tuple<id_type, point_type> coordinates_global_to_local(point_type x, int n_id = 0) const
    {
        return std::move(m::coordinates_global_to_local(inv_map(x), n_id));
    }

    id_type id(point_type x, int n_id = 0) const
    {
        return std::get<0>(m::coordinates_global_to_local(inv_map(x), n_id));
    }

    //===================================
    //

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
        m_local_min_ = m_min_;
        m_local_max_ = m_max_;
        m_memory_min_ = m_min_;
        m_memory_max_ = m_max_;


        auto dims = dimensions();

        point_type i_min, i_max;

        std::tie(i_min, i_max) = local_index_box();


        for (int i = 0; i < ndims; ++i)
        {
            ASSERT(dims[i] >= 1);

            ASSERT((m_coords_max_[i] - m_coords_min_[i] > EPSILON));

            ASSERT(i_max[i] - i_min[i] > 0);


            m_dx_[i] = (m_coords_max_[i] - m_coords_min_[i]) / static_cast<Real>(dims[i]);


            m_to_topology_scale_[i] = (i_max[i] - i_min[i]) / (m_coords_max_[i] - m_coords_min_[i]);

            m_from_topology_scale_[i] = (m_coords_max_[i] - m_coords_min_[i]) / (i_max[i] - i_min[i]);

            if (dims[i] == 1)
            {
                m_dx_[i] = 1;

                m_to_topology_scale_[i] = 0;

                m_from_topology_scale_[i] = 0;

            }

            m_to_topology_orig_[i] = i_min[i] - m_coords_min_[i] * m_to_topology_scale_[i];

            m_from_topology_orig_[i] = m_coords_min_[i] - i_min[i] * m_from_topology_scale_[i];
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



#define NOT_ZERO(_V_) ((_V_<EPSILON)?1.0:(_V_))
        m_volume_[0] = 1.0;

        m_volume_[1/* 001*/] = (dims[0] > 1) ? m_dx_[0] : 1.0;
        m_volume_[2/* 010*/] = (dims[1] > 1) ? m_dx_[1] : 1.0;
        m_volume_[4/* 100*/] = (dims[2] > 1) ? m_dx_[2] : 1.0;

//    m_volume_[1/* 001*/] = (m_dx_[0] <= EPSILON) ? 1 : m_dx_[0];
//    m_volume_[2/* 010*/] = (m_dx_[1] <= EPSILON) ? 1 : m_dx_[1];
//    m_volume_[4/* 100*/] = (m_dx_[2] <= EPSILON) ? 1 : m_dx_[2];

        m_volume_[3] /* 011 */= m_volume_[1] * m_volume_[2];
        m_volume_[5] /* 101 */= m_volume_[4] * m_volume_[1];
        m_volume_[6] /* 110 */= m_volume_[2] * m_volume_[4];
        m_volume_[7] /* 111 */= m_volume_[1] * m_volume_[2] * m_volume_[4];

        m_dual_volume_[7] = 1.0;

        m_dual_volume_[6] = m_volume_[1];
        m_dual_volume_[5] = m_volume_[2];
        m_dual_volume_[3] = m_volume_[4];

//    m_dual_volume_[6] = (m_dx_[0] <= EPSILON) ? 1 : m_dx_[0];
//    m_dual_volume_[5] = (m_dx_[1] <= EPSILON) ? 1 : m_dx_[1];
//    m_dual_volume_[3] = (m_dx_[2] <= EPSILON) ? 1 : m_dx_[2];

        m_dual_volume_[4] /* 011 */= m_dual_volume_[6] * m_dual_volume_[5];
        m_dual_volume_[2] /* 101 */= m_dual_volume_[3] * m_dual_volume_[6];
        m_dual_volume_[1] /* 110 */= m_dual_volume_[5] * m_dual_volume_[3];

        m_dual_volume_[0] /* 111 */= m_dual_volume_[6] * m_dual_volume_[5] * m_dual_volume_[3];

        m_inv_volume_[7] = 1.0;

        m_inv_volume_[1/* 001 */] = (dims[0] > 1) ? 1.0 / m_volume_[1] : 0;
        m_inv_volume_[2/* 010 */] = (dims[1] > 1) ? 1.0 / m_volume_[2] : 0;
        m_inv_volume_[4/* 100 */] = (dims[2] > 1) ? 1.0 / m_volume_[4] : 0;

        m_inv_volume_[3] /* 011 */= NOT_ZERO(m_inv_volume_[1]) * NOT_ZERO(m_inv_volume_[2]);
        m_inv_volume_[5] /* 101 */= NOT_ZERO(m_inv_volume_[4]) * NOT_ZERO(m_inv_volume_[1]);
        m_inv_volume_[6] /* 110 */= NOT_ZERO(m_inv_volume_[2]) * NOT_ZERO(m_inv_volume_[4]);
        m_inv_volume_[7] /* 111 */=
                NOT_ZERO(m_inv_volume_[1]) * NOT_ZERO(m_inv_volume_[2]) * NOT_ZERO(m_inv_volume_[4]);

        m_inv_dual_volume_[7] = 1.0;

        m_inv_dual_volume_[6/* 110 */] = (dims[0] > 1) ? 1.0 / m_dual_volume_[6] : 0;
        m_inv_dual_volume_[5/* 101 */] = (dims[1] > 1) ? 1.0 / m_dual_volume_[5] : 0;
        m_inv_dual_volume_[3/* 001 */] = (dims[2] > 1) ? 1.0 / m_dual_volume_[3] : 0;

        m_inv_dual_volume_[4] /* 011 */= NOT_ZERO(m_inv_dual_volume_[6]) * NOT_ZERO(m_inv_dual_volume_[5]);
        m_inv_dual_volume_[2] /* 101 */= NOT_ZERO(m_inv_dual_volume_[3]) * NOT_ZERO(m_inv_dual_volume_[6]);
        m_inv_dual_volume_[1] /* 110 */= NOT_ZERO(m_inv_dual_volume_[5]) * NOT_ZERO(m_inv_dual_volume_[3]);
        m_inv_dual_volume_[0] /* 111 */=
                NOT_ZERO(m_inv_dual_volume_[6]) * NOT_ZERO(m_inv_dual_volume_[5]) * NOT_ZERO(m_inv_dual_volume_[3]);
#undef NOT_ZERO
    }


    void decompose(index_tuple const &dist_dimensions, index_tuple const &dist_coord, index_type gw = 2)
    {


        index_tuple b, e;
        b = m_local_min_;
        e = m_local_max_;
        for (int n = 0; n < ndims; ++n)
        {

            m_local_min_[n] = b[n] + (e[n] - b[n]) * dist_coord[n] / dist_dimensions[n];

            m_local_max_[n] = b[n] + (e[n] - b[n]) * (dist_coord[n] + 1) / dist_dimensions[n];


            if (m_local_min_[n] == m_local_max_[n])
            {
                RUNTIME_ERROR("Mesh decompose fail! Dimension  is smaller than process grid. "
//                                "[begin= " + type_cast<std::string>(b)
//                        + ", end=" + type_cast<std::string>(e)
//                        + " ,process grid="
//                        + type_cast<std::string>(dist_coord)
                );
            }


            if (m_local_max_[n] - m_local_min_[n] > 1 && dist_dimensions[n] > 1)
            {
                m_memory_min_[n] = m_local_min_[n] - gw;
                m_memory_max_[n] = m_local_max_[n] + gw;
            }
        }


    }

};//struct StructuredMesh

} // namespace topology




typedef Topology<topology::tags::CoRectMesh> CoRectMesh;
typedef Topology<topology::tags::Curvilinear> Curvilinear;
typedef Topology<topology::tags::RectMesh> RectMesh;

template<>
struct Topology<topology::tags::CoRectMesh> : public topology::StructuredMesh
{
};

template<>
struct Topology<topology::tags::RectMesh> : public topology::StructuredMesh
{
};

template<>
struct Topology<topology::tags::Curvilinear> : public topology::StructuredMesh
{
};

} // namespace simpla

#endif //SIMPLA_TOPOLOGY_H
