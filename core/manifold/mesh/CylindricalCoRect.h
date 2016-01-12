/**
 * @file CylindricalRectMesh.h
 * @author salmon
 * @date 2016-01-11.
 */

#ifndef SIMPLA_CYLINDRICALRECTMESH_H
#define SIMPLA_CYLINDRICALRECTMESH_H

#include <limits>

#include "../mesh/MeshBlock.h"
#include "../../geometry/csCylindrical.h"
#include "../../gtl/utilities/memory_pool.h"

namespace simpla { namespace mesh
{


class CylindricalCoRect : public geometry::CylindricalMetric, public MeshBlock
{

private:
    typedef CylindricalCoRect this_type;
public:

    SP_OBJECT_HEAD(CylindricalCoRect, MeshBlock)

    typedef geometry::CylindricalMetric metric_type;

    typedef typename metric_type::cs coordinate_system_type;

    typedef MeshBlock block_type;

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

private:
    point_type m_coords_min_ = {0, 0, 0};

    point_type m_coords_max_ = {1, 1, 1};

    vector_type m_dx_ = {1, 1, 1};; //!< width of cell, except m_dx_[i]=0 when m_dims_[i]==1


public:


    CylindricalCoRect() : block_type() { }

    virtual  ~CylindricalCoRect() { }

    CylindricalCoRect(this_type const &other) = delete;

    this_type &operator=(this_type const &other) = delete;


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
    };

    box_type local_box() const
    {
        point_type l_min, l_max;

        l_min = traits::get<0>(block_type::local_index_box());

        l_max = traits::get<1>(block_type::local_index_box());


        l_min = inv_map(l_min);

        l_max = inv_map(l_max);

        return (std::make_tuple(l_min, l_max));
    }


    constexpr auto dx() const DECL_RET_TYPE(m_dx_);

    index_box_type index_box(box_type const &b) const
    {

        point_type b0, b1, x0, x1;

        b0 = traits::get<0>(block_type::local_index_box());

        b1 = traits::get<1>(block_type::local_index_box());

        x0 = traits::get<0>(b);

        x1 = traits::get<1>(b);

        if (geometry::box_intersection(b0, b1, &x0, &x1))
        {
            return std::make_tuple(
                    block_type::unpack_index(id(x0)),
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
    template<typename ...Args>
    point_type point(Args &&...args) const
    {
        return std::move(inv_map(block_type::point(std::forward<Args>(args)...)));
    }


    virtual point_type coordinates_local_to_global(std::tuple<id_type, point_type> const &t) const
    {
        return std::move(inv_map(block_type::coordinates_local_to_global(t)));
    }

    virtual std::tuple<id_type, point_type> coordinates_global_to_local(point_type x, int n_id = 0) const
    {
        return std::move(block_type::coordinates_global_to_local(map(x), n_id));
    }

    virtual id_type id(point_type const &x, int n_id = 0) const
    {
        return std::get<0>(block_type::coordinates_global_to_local(map(x), n_id));
    }



    //===================================
    //

private:
    std::vector<Real> m_volume_;
    std::vector<Real> m_dual_volume_;
    std::vector<Real> m_inv_volume_;
    std::vector<Real> m_inv_dual_volume_;

    index_type m_min_index_r_ = 0;
    index_type m_max_index_r_ = 1;

    size_t hash_(id_type s) const
    {
        return (block_type::unpack_index(s, 0) - m_min_index_r_) * block_type::NUM_OF_NODE_ID +
               block_type::node_id(s);
    }

public:
    virtual Real volume(id_type s) const
    {
        return m_volume_[hash_(s)];
    }

    virtual Real dual_volume(id_type s) const
    {
        return m_dual_volume_[hash_(s)];
    }

    virtual Real inv_volume(id_type s) const
    {

        return m_inv_volume_[hash_(s)];
    }

    virtual Real inv_dual_volume(id_type s) const
    {
        return m_inv_dual_volume_[hash_(s)];
    }

//    virtual point_type const &vertex(id_type s) const
//    {
//        return m_vertices_[base_type::hash(s) * base_type::NUM_OF_NODE_ID + base_type::sub_index(s)];
//    }


//    std::shared_ptr<point_type> m_vertices_;
public:

    virtual void deploy();

//    template<typename TGeo> void update_volume(TGeo const &geo);


    template<typename T0, typename T1, typename ...Others>
    static constexpr auto inner_product(T0 const &v0, T1 const &v1, Others &&... others)
    DECL_RET_TYPE((v0[0] * v1[0] + v0[1] * v1[1] + v0[2] * v1[2]))

};//struct RectMesh

void CylindricalCoRect::deploy()
{

//
//    if (properties().click() > this->click())
//    {
////        auto b = properties()["Geometry"]["Box"].template as<box_type>();
//
//        std::tie(m_coords_min_, m_coords_max_) = properties()["Geometry"]["Box"].template as<box_type>();
//
//
//    }

    block_type::deploy();

    auto dims = block_type::dimensions();

    {


        point_type src_min_, src_max_;

        point_type dest_min, dest_max;

        std::tie(src_min_, src_max_) = box();

        std::tie(dest_min, dest_max) = block_type::index_box();

        CHECK(dest_min);
        CHECK(dest_max);
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


    // update volume
    auto mem_index_box = block_type::memory_index_box();

    m_min_index_r_ = std::get<0>(mem_index_box)[0];
    m_max_index_r_ = std::get<1>(mem_index_box)[0];

    index_type num = (m_max_index_r_ - m_min_index_r_) * block_type::NUM_OF_NODE_ID;

    m_volume_.resize(num);
    m_inv_volume_.resize(num);
    m_dual_volume_.resize(num);
    m_inv_dual_volume_.resize(num);

    for (index_type i = m_min_index_r_; i < m_max_index_r_; ++i)
    {

        index_type s = block_type::pack(i, 0, 0);

        Real r0 = point(block_type::pack(i, 0, 0))[0];
        Real r1 = point(block_type::pack(i + 1, 0, 0))[0];
        size_t n = (i - m_min_index_r_) * block_type::NUM_OF_NODE_ID;
        //        auto dims = geo.dimensions();
//
//
#define NOT_ZERO(_V_) ((_V_<EPSILON)?1.0:(_V_))


        /**
         *\verbatim
         *                ^Z
         *               /
         *        phi   /
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
         *       000-------------001---> R
         *
         *\endverbatim
         */

        m_volume_[n + 0] = 1.0;

        m_volume_[n + 1/* 001*/] = (dims[0] > 1) ? m_dx_[0] : 1.0;
        m_volume_[n + 2/* 010*/] = (dims[1] > 1) ? m_dx_[1] : 1.0;
        m_volume_[n + 4/* 100*/] = (dims[2] > 1) ? m_dx_[2] : 1.0;


        m_volume_[n + 3] /* 011 */= m_volume_[n + 1] * m_volume_[n + 2];
        m_volume_[n + 5] /* 101 */= m_volume_[n + 4] * m_volume_[n + 1];
        m_volume_[n + 6] /* 110 */= m_volume_[n + 2] * m_volume_[n + 4];

        m_volume_[n + 7] /* 111 */= m_volume_[n + 1] * m_volume_[n + 2] * m_volume_[n + 4];

        m_volume_[n + 4 /* 100*/] *= r0;
        m_volume_[n + 6 /* 110 */] *= (r0 + r1) * 0.5;
        m_volume_[n + 7 /* 111 */] *= (r0 + r1) * 0.5;

        m_dual_volume_[n + 7] = 1.0;

        m_dual_volume_[n + 6/* 110 */] = m_volume_[n + 1];
        m_dual_volume_[n + 5/* 101 */] = m_volume_[n + 2];
        m_dual_volume_[n + 3/* 011 */] = m_volume_[n + 4];

        m_dual_volume_[n + 4] /* 011 */= m_dual_volume_[n + 6] * m_dual_volume_[n + 5];
        m_dual_volume_[n + 2] /* 101 */= m_dual_volume_[n + 3] * m_dual_volume_[n + 6];
        m_dual_volume_[n + 1] /* 110 */= m_dual_volume_[n + 5] * m_dual_volume_[n + 3];

        m_dual_volume_[n + 0] /* 111 */= m_dual_volume_[n + 6] * m_dual_volume_[n + 5] * m_dual_volume_[n + 3];


        m_dual_volume_[n + 1] *= (r0 + r1) * 0.5;
        m_dual_volume_[n + 4] *= r0;
        m_dual_volume_[n + 0] *= r0;


        m_inv_volume_[n + 7] = 1.0;

        m_inv_volume_[n + 1/* 001 */] = (dims[0] > 1) ? 1.0 / m_volume_[n + 1] : 0;
        m_inv_volume_[n + 2/* 010 */] = (dims[1] > 1) ? 1.0 / m_volume_[n + 2] : 0;
        m_inv_volume_[n + 4/* 100 */] = (dims[2] > 1) ? 1.0 / m_volume_[n + 4] : 0;

        m_inv_volume_[n + 3] /* 011 */= NOT_ZERO(m_inv_volume_[n + 1]) * NOT_ZERO(m_inv_volume_[n + 2]);
        m_inv_volume_[n + 5] /* 101 */= NOT_ZERO(m_inv_volume_[n + 4]) * NOT_ZERO(m_inv_volume_[n + 1]);
        m_inv_volume_[n + 6] /* 110 */= NOT_ZERO(m_inv_volume_[n + 2]) * NOT_ZERO(m_inv_volume_[n + 4]);
        m_inv_volume_[n + 7] /* 111 */=
                NOT_ZERO(m_inv_volume_[n + 1]) * NOT_ZERO(m_inv_volume_[n + 2]) * NOT_ZERO(m_inv_volume_[n + 4]);


        m_inv_volume_[n + 4 /* 100*/] /= r0;
        m_inv_volume_[n + 6 /* 110 */] /= (r0 + r1) * 0.5;
        m_inv_volume_[n + 7 /* 111 */] /= (r0 + r1) * 0.5;

        m_inv_dual_volume_[n + 7] = 1.0;

        m_inv_dual_volume_[n + 6/* 110 */] = (dims[0] > 1) ? 1.0 / m_dual_volume_[n + 6] : 0;
        m_inv_dual_volume_[n + 5/* 101 */] = (dims[1] > 1) ? 1.0 / m_dual_volume_[n + 5] : 0;
        m_inv_dual_volume_[n + 3/* 001 */] = (dims[2] > 1) ? 1.0 / m_dual_volume_[n + 3] : 0;

        m_inv_dual_volume_[n + 4] /* 011 */= NOT_ZERO(m_inv_dual_volume_[n + 6]) * NOT_ZERO(m_inv_dual_volume_[n + 5]);
        m_inv_dual_volume_[n + 2] /* 101 */= NOT_ZERO(m_inv_dual_volume_[n + 3]) * NOT_ZERO(m_inv_dual_volume_[n + 6]);
        m_inv_dual_volume_[n + 1] /* 110 */= NOT_ZERO(m_inv_dual_volume_[n + 5]) * NOT_ZERO(m_inv_dual_volume_[n + 3]);
        m_inv_dual_volume_[n + 0] /* 111 */=
                NOT_ZERO(m_inv_dual_volume_[n + 6]) * NOT_ZERO(m_inv_dual_volume_[n + 5]) *
                NOT_ZERO(m_inv_dual_volume_[n + 3]);


        m_inv_dual_volume_[n + 1] /= (r0 + r1) * 0.5;
        m_inv_dual_volume_[n + 4] /= r0;
        m_inv_dual_volume_[n + 0] /= r0;
#undef NOT_ZERO

    }


}

}}  // namespace mesh // namespace simpla
#endif //SIMPLA_CYLINDRICALRECTMESH_H
