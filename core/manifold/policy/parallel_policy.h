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
namespace manifold { namespace policy
{
/**
 * @ingroup manifold
 */
template<typename ...> struct ParallelPolicy;

template<typename TMesh>
struct ParallelPolicy<TMesh>
{

private:
    typedef TMesh mesh_type;

    typedef ParallelPolicy<mesh_type> this_type;


public:

    typedef this_type parallel_policy;

    ParallelPolicy(mesh_type &geo) : m_mesh_(geo) { }

    virtual ~ParallelPolicy() { }

    template<typename TDict> void load(TDict const &dict) { }


    template<typename OS>
    OS &print(OS &os) const
    {
        os << "\t ParallelPolicy={ Default }," << std::endl;
        return os;
    }


    void deploy();

    void sync(DataSet &) const;


    template<typename T> void sync(T &self) const;


    template<int IFORM, typename Func, typename ...Args>
    void update(Func const &fun, Args &&...args) const;


    template<typename TV, int IFORM, typename Range, typename Func>
    void for_each1(DataSet &ds, Range const &r, Func const &fun);

    template<typename TV, int IFORM, typename Range, typename Func>
    void for_each1(DataSet const &ds, Range const &r, Func const &fun) const;


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


}; //template<typename TMesh> struct ParallelPolicy




template<typename TMesh>
void ParallelPolicy<TMesh>::deploy()
{

    if (GLOBAL_COMM.is_valid())
    {
        m_mesh_.decompose(GLOBAL_COMM.topology(),
                          GLOBAL_COMM.coordinate());
    }
}

template<typename TMesh> void
ParallelPolicy<TMesh>::sync(DataSet &ds) const
{
    parallel::sync(ds);
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
    if (m_mesh_.boundary_box().size() > 0)
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
template<typename TV, int IFORM, typename TRange, typename Func>
void  ParallelPolicy<TMesh>::for_each1(DataSet const &ds, TRange const &r0, Func const &fun) const
{

    serial::parallel_for(
            r0,
            [&](TRange const &r)
            {
                for (auto const &item:r)
                {
                    fun(item, reinterpret_cast<TV *>(ds.data.get())[m_mesh_.hash(item)]);
                }
            }
    );
}

template<typename TMesh>
template<typename TV, int IFORM, typename TRange, typename Func>
void  ParallelPolicy<TMesh>::for_each1(DataSet &ds, TRange const &r0, Func const &fun)
{

    serial::parallel_for(
            r0,
            [&](TRange const &r)
            {
                for (auto const &item:r)
                {
                    fun(item, reinterpret_cast<TV *>(ds.data.get())[m_mesh_.hash(item)]);
                }
            }
    );
}

template<typename TMesh>
template<int IFORM, typename Func>
void  ParallelPolicy<TMesh>::for_each_boundary(Func const &fun) const
{
    for (auto const &item:m_mesh_.boundary_box())
    {
        parallel::parallel_for(m_mesh_.template make_range<IFORM>(item), fun);
    }
};

template<typename TMesh>
template<int IFORM, typename Func>
void  ParallelPolicy<TMesh>::for_each_ghost(Func const &fun) const
{
    for (auto const &item:m_mesh_.ghost_box())
    {
        parallel::parallel_for(m_mesh_.template make_range<IFORM>(item), fun);
    }
};

template<typename TMesh>
template<int IFORM, typename Func>
void  ParallelPolicy<TMesh>::for_each_center(Func const &fun) const
{
    parallel::parallel_for(m_mesh_.template make_range<IFORM>(m_mesh_.center_box()), fun);
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
