/**
 * @file MeshBlockStructured.h
 * @author salmon
 * @date 2016-05-18.
 */

#ifndef SIMPLA_MESH_MESHBLOCKSTRUCTURED_H
#define SIMPLA_MESH_MESHBLOCKSTRUCTURED_H

#include "../../base/Object.h"
#include "../../gtl/IteratorBlock.h"
#include "../../gtl/Log.h"
#include "../Mesh.h"
#include "../MeshEntityIdCoder.h"

namespace simpla { namespace tags { struct split; }}

namespace simpla { namespace mesh
{
namespace tags { template<int> class BlockStructured; }
/**
 *  MeshBlockStructured : Block Structured Mesh
 *  - topology  : rectangle or hexahedron;
 *  - cell size : uniform structured;
 *  - decompose, refine,coarse => Block Structured Mesh
 *
 */
template<int NDIMS>
class Mesh<tags::BlockStructured<NDIMS>> : public base::Object
{
    typedef Mesh<tags::BlockStructured<NDIMS>> this_type;
public:

// types
    static constexpr int ndims = NDIMS;
    typedef size_t size_type;
    typedef gtl::IteratorBlock<size_type, ndims> iterator;
private:
    point_type m_lower_{0, 0, 0};
    point_type m_upper_{1, 1, 1};
    index_tuple m_dims_{1, 1, 1};
    index_tuple m_gw_{0, 0, 0}; // gw=0 mean cycle coordinate

    vector_type m_dx_{1, 1, 1};
    index_tuple m_offset_{0, 0, 0};
    index_tuple m_shape_{1, 1, 1};


public:

    Mesh(point_type const &lower, point_type const &upper, index_tuple const &dims,
         index_tuple const *ghost_width = nullptr) :
            m_lower_(lower), m_upper_(upper), m_dims_(dims)
    {
        if (ghost_width != nullptr) { m_gw_ = (ghost_width); }

        for (int i = 0; i < ndims; ++i)
        {
            if (m_dims_[i] <= 1) { m_dims_[i] = 1; }

            if (m_gw_[i] * 2 > m_dims_[i])
            {
                RUNTIME_ERROR << "Illegal mesh shape! [ dims=" << m_dims_ << " , ghost width = " << m_gw_ << "]" <<
                std::endl;
            }
        }

        m_dx_ = (m_upper_ - m_lower_) / m_dims_;

        m_offset_ = m_gw_;

        m_shape_ = m_dims_ + m_offset_ + m_gw_;

    }

    Mesh(this_type const &other)
            : m_lower_(other.m_lower_), m_upper_(other.m_upper_), m_dims_(other.m_dims_),
              m_dx_(other.m_dx_), m_gw_(other.m_gw_), m_offset_(other.m_offset_), m_shape_(other.m_shape_)
    {

    }

    Mesh(this_type &&other)
            : m_lower_(other.m_lower_), m_upper_(other.m_upper_), m_dims_(other.m_dims_),
              m_dx_(other.m_dx_), m_gw_(other.m_gw_), m_offset_(other.m_offset_), m_shape_(other.m_shape_)
    {
    }

    void swap(this_type &other)
    {
        std::swap(m_lower_, other.m_lower_);
        std::swap(m_upper_, other.m_upper_);
        std::swap(m_dims_, other.m_dims_);
        std::swap(m_dx_, other.m_dx_);
        std::swap(m_gw_, other.m_gw_);
        std::swap(m_offset_, other.m_offset_);
        std::swap(m_shape_, other.m_shape_);
    }

    this_type &operator=(this_type const &other)
    {
        this_type(other).swap(*this);
        return *this;
    }

    ~Mesh() { }



    //****************************************

    int number_of_dims() const { return ndims; }

    index_tuple const &dimensions() const { return m_dims_; }

    std::tuple<index_tuple, index_tuple, index_tuple> shape() const
    {
        return std::make_tuple(m_shape_, m_offset_, m_dims_);
    }

    index_tuple const &ghost_width() const { return m_gw_; }

    bool is_periodic(int n) const { return m_gw_[n % 3] == 0; }

    box_type boundary_box() const { return std::make_pair(m_lower_, m_upper_); }

    //****************************************

    MeshEntityRange range() const;

    //****************************************


    /**
     *  remove periodic axis, which  ghost_width==0
     */
    id_type periodic_id_mask() const
    {
        id_type M0 = ((1UL << ID_DIGITS) - 1);
        id_type M1 = ((1UL << (MESH_RESOLUTION)) - 1);
        return FULL_OVERFLOW_FLAG
               | (is_periodic(0) ? M1 : M0)
               | ((is_periodic(1) ? M1 : M0) << ID_DIGITS)
               | ((is_periodic(2) ? M1 : M0) << (ID_DIGITS * 2));
    }

    size_t id_mask() const
    {
        id_type M0 = ((1UL << ID_DIGITS) - 1);
        id_type M1 = ((1UL << (MESH_RESOLUTION)) - 1);
        return FULL_OVERFLOW_FLAG
               | ((m_idx_max_[0] - m_idx_min_[0] > 1) ? M0 : M1)
               | (((m_idx_max_[1] - m_idx_min_[1] > 1) ? M0 : M1) << ID_DIGITS)
               | (((m_idx_max_[2] - m_idx_min_[2] > 1) ? M0 : M1) << (ID_DIGITS * 2));
    }


    index_box_type index_box() const
    {
        return std::make_tuple(m_idx_min_, m_idx_max_);
    }


//    index_box_type cell_index_box(id_type const &s) const
//    {
//        return std::make_tuple(m::unpack_index(s - _DA), m::unpack_index(s + _DA));
//    }
    index_box_type index_box(box_type const &b) const
    {

        point_type b0, b1, x0, x1;

        std::tie(b0, b1) = local_index_box();

        std::tie(x0, x1) = b;

        if (geometry::box_intersection(b0, b1, &x0, &x1))
        {
            return std::make_tuple(m::unpack_index(id(x0)),
                                   m::unpack_index(id(x1) + (m::_DA << 1)));

        }
        else
        {
            index_tuple i0, i1;
            i0 = 0;
            i1 = 0;
            return std::make_tuple(i0, i1);
        }

    }

    index_box_type local_index_box() const
    {
        return (std::make_tuple(m_idx_local_min_, m_idx_local_max_));
    }

    index_box_type memory_index_box() const
    {
        return (std::make_tuple(m_idx_memory_min_, m_idx_memory_max_));
    }

    bool in_box(index_tuple const &x) const
    {
        return (m_idx_local_min_[1] <= x[1]) && (m_idx_local_min_[2] <= x[2]) && (m_idx_local_min_[0] <= x[0])  //
               && (m_idx_local_max_[1] > x[1]) && (m_idx_local_max_[2] > x[2]) && (m_idx_local_max_[0] > x[0]);

    }

    bool in_box(id_type s) const { return in_box(m::unpack_index(s)); }

    template<int IFORM>
    range_type range() const
    {
        return m::template make_range<IFORM>(m_idx_local_min_, m_idx_local_max_);
    }

    template<int IFORM>
    range_type inner_range() const
    {
        index_tuple i_min = m_idx_local_min_;
        index_tuple i_max = m_idx_local_max_;
        for (int n = 0; n < ndims; ++n)
        {
            if (i_max[n] - i_min[n] > 2 * m_idx_min_[n])
            {
                i_min[n] += m_idx_min_[n];
                i_max[n] -= m_idx_min_[n];
            }

        }
        return m::template make_range<IFORM>(i_min, i_max);
    }

    template<int IFORM>
    range_type outer_range() const
    {
        return m::template make_range<IFORM>(m_idx_memory_min_, m_idx_memory_max_);
    }


    template<int IFORM>
    size_t max_hash() const
    {
        return static_cast<size_t>(m::hash(
                m::pack_index(m_idx_memory_max_ - 1, m::template sub_index_to_id<IFORM>(3UL)),
                m_idx_memory_min_, m_idx_memory_max_));
    }


    size_t hash(id_type const &s) const
    {
        return static_cast<size_t>(m::hash(s, m_idx_memory_min_, m_idx_memory_max_));
    }


    index_box_type const &center_box() const { return m_center_box_; }

    std::vector<index_box_type> const &boundary_box() const { return m_boundary_box_; }

    std::vector<index_box_type> const &ghost_box() const { return m_ghost_box_; }




    //================================================================================================
    // @name Coordinates dependent

public:

    void box(box_type const &b) { std::tie(m_min_, m_max_) = b; }

    box_type box() const { return (std::make_tuple(m_min_, m_max_)); }

    vector_type const &dx() const { return m_dx_; }

    box_type cell_box(id_type const &s) const
    {
        return std::make_tuple(point(s - m::_DA), point(s + m::_DA));
    }

    int get_vertices(size_t node_id, id_type s, point_type *p = nullptr) const
    {

        int num = m::get_adjacent_cells(VERTEX, node_id, s);

        if (p != nullptr)
        {
            id_type neighbour[num];

            m::get_adjacent_cells(VERTEX, node_id, s, neighbour);

            for (int i = 0; i < num; ++i)
            {
                p[i] = point(neighbour[i]);
            }

        }

        return num;
    }

    void get_volumes(Real *v, Real *inv_v, Real *dual_v, Real *inv_dual_v);

private:
    /**
     * @name  Coordinate map
     * @{
     *
     *        Topology mesh       geometry mesh
     *                        map
     *              M      ---------->      G
     *              x                       y
     **/

    point_type m_l2g_shift_ = {0, 0, 0};

    point_type m_l2g_scale_ = {1, 1, 1};

    point_type m_g2l_shift_ = {0, 0, 0};

    point_type m_g2l_scale_ = {1, 1, 1};


    point_type inv_map(point_type const &x) const
    {

        point_type res;

        res[0] = std::fma(x[0], m_g2l_scale_[0], m_g2l_shift_[0]);

        res[1] = std::fma(x[1], m_g2l_scale_[1], m_g2l_shift_[1]);

        res[2] = std::fma(x[2], m_g2l_scale_[2], m_g2l_shift_[2]);

        return std::move(res);
    }

    point_type map(point_type const &y) const
    {

        point_type res;


        res[0] = std::fma(y[0], m_l2g_scale_[0], m_l2g_shift_[0]);

        res[1] = std::fma(y[1], m_l2g_scale_[1], m_l2g_shift_[1]);

        res[2] = std::fma(y[2], m_l2g_scale_[2], m_l2g_shift_[2]);

        return std::move(res);
    }

public:

    virtual point_type point(id_type const &s) const
    {
        return std::move(map(m::point(s)));
    }

    virtual point_type coordinates_local_to_global(id_type s, point_type const &x) const
    {
        return std::move(map(m::coordinates_local_to_global(s, x)));
    }

    virtual point_type coordinates_local_to_global(std::tuple<id_type, point_type> const &t) const
    {
        return std::move(map(m::coordinates_local_to_global(t)));
    }

    virtual std::tuple<id_type, point_type> coordinates_global_to_local(point_type const &x,
                                                                        int n_id = 0) const
    {
        return std::move(m::coordinates_global_to_local(inv_map(x), n_id));
    }

    virtual id_type id(point_type const &x, int n_id = 0) const
    {
        return std::get<0>(m::coordinates_global_to_local(inv_map(x), n_id));
    }


};//class MeshBlockStructured


template<int NDIMS>
class Mesh<tags::BlockStructured<NDIMS>>::View
{
public:
    MeshEntityRange range() const;
};

template<int NDIMS>
class Mesh<tags::BlockStructured<NDIMS>>::View::Iterator
{

};
}}//namespace simpla { namespace mesh
#endif //SIMPLA_MESH_MESHBLOCKSTRUCTURED_H
