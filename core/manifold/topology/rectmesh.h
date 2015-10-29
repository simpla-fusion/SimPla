/**
 * @file rect_mesh.h
 * @author salmon
 * @date 2015-10-27.
 */

#ifndef SIMPLA_RECT_MESH_H
#define SIMPLA_RECT_MESH_H

#include "mesh_block.h"

namespace simpla { namespace topology
{


template<typename TMap = LinearMap3>
struct RectMesh : public MeshBlock, public TMap
{

private:
    typedef RectMesh<TMap> this_type;
    typedef MeshBlock base_type;
    typedef TMap map_type;
public:
    using base_type::id_type;
    using base_type::id_tuple;
    using base_type::index_type;
    typedef id_type value_type;
    typedef size_t difference_type;
    typedef nTuple<Real, ndims> point_type;
    typedef nTuple<Real, ndims> vector_type;
    using base_type::index_tuple;

private:
    point_type m_coords_min_ = {0, 0, 0};

    point_type m_coords_max_ = {1, 1, 1};

    vector_type m_dx_ = {1, 1, 1};; //!< width of cell, except m_dx_[i]=0 when m_dims_[i]==1

    bool m_is_valid_ = false;
public:


    RectMesh() : base_type()
    {

    }


    RectMesh(this_type const &other) :
            base_type(other), m_dx_(other.m_dx_)
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

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }


    template<typename TDict>
    void load(TDict const &dict)
    {
        box(dict["Geometry"]["Box"].template as<std::tuple<point_type, point_type> >());

        base_type::dimensions(
                dict["Geometry"]["Topology"]["Dimensions"].template as<index_tuple>(index_tuple{10, 1, 1}));
    }

    template<typename OS>
    OS &print(OS &os) const
    {

        os << "\t\tTopology = {" << std::endl
        << "\t\t Type = \"RectMesh\",  }," << std::endl
        << "\t\t Extents = {" << box() << "}," << std::endl
        << "\t\t Dimensions = " << base_type::dimensions() << "," << std::endl
        << "\t\t}, " << std::endl;

        return os;
    }


    bool is_valid() const { return m_is_valid_; }

    //================================================================================================
    // @name Coordinates dependent



    point_type epsilon() const { return EPSILON * m_from_topology_scale_; }

    template<typename X0, typename X1>
    void box(X0 const &x0, X1 const &x1)
    {
        m_coords_min_ = x0;
        m_coords_max_ = x1;
    }

    template<typename T0> void box(T0 const &b)
    {
        box(simpla::traits::get<0>(b), simpla::traits::get<1>(b));
    }

    std::tuple<point_type, point_type> box() const
    {
        return (std::make_tuple(m_coords_min_, m_coords_max_));
    }

    std::tuple<point_type, point_type> local_box() const
    {
        point_type l_min, l_max;

        std::tie(l_min, l_max) = base_type::local_box();

        l_min = inv_map(l_min);

        l_max = inv_map(l_max);

        return (std::make_tuple(l_min, l_max));
    }

    constexpr auto dx() const DECL_RET_TYPE(m_dx_);

private:
    using map_type::map;
    using map_type::inv_map;
public:
    virtual point_type point(id_type const &s) const { return std::move(map(base_type::point(s))); }

/**
 * @bug: truncation error of coordinates transform larger than 1000
 *     epsilon (1e4 epsilon for cylindrical coordinates)
 * @param args
 * @return
 *
 */
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

    //===================================
    //

    virtual Real volume(id_type s) const { return m_volume_.get()[hash((s & (~m::_DA)) << 1)]; }

    virtual Real dual_volume(id_type s) const { return m_dual_volume_.get()[hash((s & (~m::_DA)) << 1)]; }

    virtual Real inv_volume(id_type s) const { return m_inv_volume_.get()[hash((s & (~m::_DA)) << 1)]; }

    virtual Real inv_dual_volume(id_type s) const { return m_inv_dual_volume_.get()[hash((s & (~m::_DA)) << 1)]; }

    virtual point_type const &vertex(id_type s) const { return m_vertics_.get()[hash(s & (~m::_DA))]; }


private:
    std::shared_ptr<Real> m_volume_;
    std::shared_ptr<Real> m_dual_volume_;
    std::shared_ptr<Real> m_inv_volume_;
    std::shared_ptr<Real> m_inv_dual_volume_;
    std::shared_ptr<point_type> m_vertics_;
public:

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


        m_is_valid_ = true;
    }

};//struct RectMesh

}}  // namespace topology // namespace simpla

#endif //SIMPLA_RECT_MESH_H
