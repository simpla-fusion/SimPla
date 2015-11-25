/**
 * @file parallel_policy.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_PARALLEL_POLICY_H
#define SIMPLA_PARALLEL_POLICY_H

#include "../../parallel/mpi_comm.h"
#include "../../parallel/mpi_update.h"
#include "../../parallel/parallel.h"
#include "../../parallel/distributed_object.h"

namespace simpla
{
/**
 * @ingroup manifold
 */
namespace manifold
{
namespace policy
{


template<typename ...>
struct ParallelPolicy;

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


//    template<int IFORM, typename Func>
//    void update(Func const &fun) const;

    template<int IFORM, typename Func, typename ...Args>
    void update(Func const &fun, Args &&...args) const;

private:

    mesh_type &m_mesh_;

    std::tuple<nTuple<size_t, 3>, nTuple<size_t, 3>> m_center_box_;

    std::vector<std::tuple<nTuple<size_t, 3>, nTuple<size_t, 3>>> m_boundary_box_;


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

}

template<typename TMesh>
template<int IFORM, typename Func, typename ...Args>
void ParallelPolicy<TMesh>::update(Func const &fun, Args &&...args) const
{
    if (m_boundary_box_.size() > 0)
    {
        for (auto const &item:m_boundary_box_)
        {
            parallel::parallel_for(m_mesh_.template make_range<IFORM>(
                    std::get<0>(item), std::get<1>(item)), fun);

        }

        parallel::DistributedObject dist_obj;

        dist_obj.add(std::forward<Args>(args)...);

        dist_obj.sync();

        parallel::parallel_for(
                m_mesh_.template make_range<IFORM>(std::get<0>(m_center_box_), std::get<1>(m_center_box_)), fun);

        dist_obj.wait();
    }
    else
    {
        parallel::parallel_for(m_mesh_.template range<IFORM>(), fun);
    }
}


template<typename TMesh>
template<typename T> void
ParallelPolicy<TMesh>::sync(T &self) const
{
    parallel::sync(traits::get_dataset(self));
}

} //namespace policy
} //namespace manifold
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
