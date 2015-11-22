/**
 * @file amr.h
 * @author salmon
 * @date 2015-11-20.
 */

#ifndef SIMPLA_AMR_H
#define SIMPLA_AMR_H

#include <signal.h>

namespace simpla { namespace manifold { namespace policy
{


template<typename TMesh>
struct AMRPolicy
{
private:
    typedef AMRPolicy<TMesh> this_type;
public:
    typedef TMesh mesh_type;
    typedef typename mesh_type::point_type point_type;
    typedef typename mesh_type::id_type id_type;
    typedef typename mesh_type::index_tuple index_tuple;
    typedef typename mesh_type::patch_type patch_type;

    typedef this_type amr_policy;

private:
    mesh_type &m_mesh_;
public:

    patch_type coarsen(index_tuple const &min, index_tuple const &min);

    patch_type refine();


private:
    signal m_coarsen_

};


}}}//namespace simpla{namespace manifold{namespace policy

#endif //SIMPLA_AMR_H
