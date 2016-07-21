/**
 * @file parallel_policy.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_PARALLEL_POLICY_H
#define SIMPLA_PARALLEL_POLICY_H

#include <tuple>
#include <vector>
#include "../../gtl/nTuple.h"
#include "../../parallel/MPIComm.h"
#include "../../parallel/MPIUpdate.h"
#include "../../parallel/Parallel.h"
#include "../../parallel/DistributedObject.h"

namespace simpla { namespace manifold { namespace policy
{
/**
 * @ingroup CoordinateSystem
 */

template<typename TMesh>
struct ParallelPolicy
{

private:
    typedef TMesh mesh_type;

    typedef ParallelPolicy<mesh_type> this_type;


public:

    typedef this_type parallel_policy;

    ParallelPolicy(mesh_type &geo) : m_mesh_(geo) { }

    virtual ~ParallelPolicy() { }

    template<typename TDict> void load(TDict const &dict) { }


    virtual std::ostream &print(std::ostream &os) const
    {
        os << "\t ParallelPolicy={ Default }," << std::endl;
        return os;
    }


    void deploy();

    void sync(data_model::DataSet &) const;


    template<typename T> void sync(T &self) const;


    template<int IFORM, typename Func, typename ...Args>
    void update(Func const &fun, Args &&...args) const;


    template<typename TV, int IFORM, typename TF, typename Range, typename Func>
    void for_each_value(TF &f, Range const &r, Func const &fun) const;


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
        m_mesh_.decompose(GLOBAL_COMM.topology_dims(),
                          GLOBAL_COMM.topology_coordinate());
    }
}

template<typename TMesh> void
ParallelPolicy<TMesh>::sync(data_model::DataSet &ds) const
{
    parallel::sync(ds);
}

template<typename TMesh>
template<typename T> void
ParallelPolicy<TMesh>::sync(T &self) const
{
    parallel::sync(data_model::DataSet::create(self));
}

template<typename TMesh>
template<int IFORM, typename Func, typename ...Args>
void ParallelPolicy<TMesh>::update(Func const &fun, Args &&...args) const
{
    if (m_mesh_.boundary_box().size() > 0)
    {
        for_each_boundary<IFORM>(fun);

        parallel::DistributedObject dist_obj;

        dist_obj.add(std::forward<Args>(args)..., <#initializer#>, false);

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

//template<typename TMesh>
//template<typename TV, int IFORM, typename TF, typename TRange, typename Func>
//void  ParallelPolicy<TMesh>::for_each1(TF const &f, TRange const &r0, Func const &fun) const
//{
//
//    parallel::parallel_for(
//            r0,
//            [&](TRange const &r)
//            {
//                for (auto const &s:r)
//                {
//                    fun(s, f.at(s));
//                }
//            }
//    );
//}

template<typename TMesh>
template<typename TV, int IFORM, typename TF, typename TRange, typename Func>
void  ParallelPolicy<TMesh>::for_each_value(TF &f, TRange const &r0, Func const &fun) const
{

    parallel::parallel_for(
            r0,
            [&](TRange const &r)
            {
                for (auto const &s:r)
                {
                    fun(s, f.at(s));
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


}}}// namespace simpla { namespace CoordinateSystem { namespace policy



#endif //SIMPLA_PARALLEL_POLICY_H
