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

            m_memory_max_(other.m_memory_max_)
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

    }

    template<typename TDict>
    void load(TDict const &dict)
    {
        m_min_ = 0;
        m_max_ = m_min_ + dict["Topology"]["Dimensions"].template as<id_tuple>();
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


    void deploy()
    {
        m_local_min_ = m_min_;
        m_local_max_ = m_max_;
        m_memory_min_ = m_min_;
        m_memory_max_ = m_max_;
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
                RUNTIME_ERROR(
                        "Mesh decompose fail! Dimension  is smaller than process grid. "
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
    void box(T0 const &min, T1 const &max)
    {
        m_min_ = min;
        m_max_ = max;
    };


    auto box() const
    DECL_RET_TYPE((std::forward_as_tuple(m_min_, m_max_)))


    auto local_box() const
    DECL_RET_TYPE((std::forward_as_tuple(m_local_min_, m_local_max_)))

    auto memory_box() const
    DECL_RET_TYPE((std::forward_as_tuple(m_memory_min_, m_memory_max_)))


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


    size_t hash(id_type const &s) const { return m::hash(s, m_memory_min_, m_memory_max_); }

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
