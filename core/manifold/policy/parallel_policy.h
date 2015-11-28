/**
 * @file parallel_policy.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_PARALLEL_POLICY_H
#define SIMPLA_PARALLEL_POLICY_H

#include <tuple>
#include <vector>
#include "../../gtl/ntuple.h"
#include "../../parallel/mpi_comm.h"
#include "../../parallel/mpi_update.h"
#include "../../parallel/parallel.h"
#include "../../parallel/distributed_object.h"

namespace simpla
{

/**
 * @ingroup manifold
 */
namespace manifold { namespace policy
{

template<typename ...> struct ParallelPolicy;

template<typename TMesh>
struct ParallelPolicy<TMesh>
{

private:
    typedef TMesh mesh_type;

    typedef ParallelPolicy<mesh_type> this_type;


public:

    typedef this_type parallel_policy;

    ParallelPolicy(mesh_type &geo) :
            m_mesh_(geo)
    {
    }

    virtual ~ParallelPolicy()
    {
    }

    template<typename TDict>
    void load(TDict const &dict)
    {
    }


    template<typename OS>
    OS &print(OS &os) const
    {
        os << "\t ParallelPolicy={ Default }," << std::endl;
        return os;
    }


    void deploy();


    template<typename T> void sync(T &self) const;


    template<int IFORM, typename Func, typename ...Args>
    void update(Func const &fun, Args &&...args) const;

    template<int IFORM, typename Func>
    void for_each(Func const &fun) const;

    template<int IFORM, typename Func>
    void for_each_boundary(Func const &fun) const;

    template<int IFORM, typename Func>
    void for_each_ghost(Func const &fun) const;

    template<int IFORM, typename Func>
    void for_each_center(Func const &fun) const;

private:

    mesh_type &m_mesh_;
    typedef nTuple<size_t, 3> IVec3;

    std::tuple<IVec3, IVec3> m_center_box_;
    std::vector<std::tuple<IVec3, IVec3>> m_boundary_box_;
    std::vector<std::tuple<IVec3, IVec3>> m_ghost_box_;


}; //template<typename TMesh> struct ParallelPolicy




template<typename TMesh>
void ParallelPolicy<TMesh>::deploy()
{
    auto &m_mpi_comm_ = SingletonHolder<parallel::MPIComm>::instance();

    if (m_mpi_comm_.is_valid())
    {
        m_mesh_.decompose(m_mpi_comm_.topology(), m_mpi_comm_.coordinate());
    }

    nTuple<size_t, mesh_type::ndims> m_min, m_max;
    nTuple<size_t, mesh_type::ndims> l_min, l_max;
    nTuple<size_t, mesh_type::ndims> c_min, c_max;
    nTuple<size_t, mesh_type::ndims> ghost_width;
    std::tie(m_min, m_max) = m_mesh_.memory_index_box();
    std::tie(l_min, l_max) = m_mesh_.local_index_box();
    c_min = l_min + (l_min - m_min);
    c_max = l_max - (m_max - l_max);
    m_center_box_ = std::make_tuple(c_min, c_max);

    for (int i = 0; i < mesh_type::ndims; ++i)
    {
        nTuple<size_t, mesh_type::ndims> b_min, b_max;

        b_min = l_min;
        b_max = l_max;
        b_max[i] = c_min[i];
        if (b_min[i] != b_max[i]) { m_boundary_box_.push_back(std::make_tuple(b_min, b_max)); }
        b_min = l_min;
        b_max = l_max;
        b_min[i] = c_max[i];
        if (b_min[i] != b_max[i]) { m_boundary_box_.push_back(std::make_tuple(b_min, b_max)); }
        l_min[i] = c_min[i];
        l_max[i] = c_max[i];
    }

    std::tie(m_min, m_max) = m_mesh_.memory_index_box();
    std::tie(l_min, l_max) = m_mesh_.local_index_box();

    for (int i = 0; i < mesh_type::ndims; ++i)
    {
        nTuple<size_t, mesh_type::ndims> g_min, g_max;


        g_min = m_min;
        g_max = m_max;
        g_min[i] = m_min[i];
        g_max[i] = l_min[i];
        if (g_min[i] != g_max[i]) { m_ghost_box_.push_back(std::make_tuple(g_min, g_max)); }
        g_min = g_min;
        g_max = g_max;

        g_min[i] = l_max[i];
        g_max[i] = m_max[i];
        if (g_min[i] != g_max[i]) { m_ghost_box_.push_back(std::make_tuple(g_min, g_max)); }
        m_min[i] = l_min[i];
        m_max[i] = l_max[i];
    }

}

template<typename TMesh>
template<typename T> void
ParallelPolicy<TMesh>::sync(T &self) const
{
    parallel::sync(traits::make_dataset(self));
}

template<typename TMesh>
template<int IFORM, typename Func, typename ...Args>
void ParallelPolicy<TMesh>::update(Func const &fun, Args &&...args) const
{
    if (m_boundary_box_.size() > 0)
    {
        for_each_boundary<IFORM>(fun);

        parallel::DistributedObject dist_obj;

        dist_obj.add(std::forward<Args>(args)...);

        dist_obj.sync();

        for_each_center<IFORM>(fun);

        dist_obj.wait();
    }
    else
    {
        for_each<IFORM>(fun);
    }
}

template<typename TMesh>
template<int IFORM, typename Func>
void  ParallelPolicy<TMesh>::for_each(Func const &fun) const
{
    parallel::parallel_for(m_mesh_.template range<IFORM>(), fun);
};

template<typename TMesh>
template<int IFORM, typename Func>
void  ParallelPolicy<TMesh>::for_each_boundary(Func const &fun) const
{
    for (auto const &item:m_boundary_box_)
    {
        parallel::parallel_for(m_mesh_.template make_range<IFORM>(item), fun);
    }
};

template<typename TMesh>
template<int IFORM, typename Func>
void  ParallelPolicy<TMesh>::for_each_ghost(Func const &fun) const
{
    for (auto const &item:m_ghost_box_)
    {
        parallel::parallel_for(m_mesh_.template make_range<IFORM>(item), fun);
    }
};

template<typename TMesh>
template<int IFORM, typename Func>
void  ParallelPolicy<TMesh>::for_each_center(Func const &fun) const
{
    parallel::parallel_for(m_mesh_.template make_range<IFORM>(m_center_box_), fun);
};


}} //namespace policy  //namespace manifold


namespace traits
{

template<typename TMesh>
struct type_id<manifold::policy::ParallelPolicy<TMesh>>
{
    static std::string name()
    {
        return "ParallelPolicy<" + type_id<TMesh>::name() + ">";
    }
};
}

}//namespace simpla
#endif //SIMPLA_PARALLEL_POLICY_H
