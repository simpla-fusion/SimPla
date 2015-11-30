/**
 * @file constraint.h
 * @author salmon
 * @date 2015-11-29.
 */

#ifndef SIMPLA_CONSTRAINT_H
#define SIMPLA_CONSTRAINT_H

#include <memory>
#include "../parallel/parallel.h"
#include "../gtl/primitives.h"

namespace simpla
{

template<typename M> using surface_constraint= typedef parallel::concurrent_unordered_map<typename M::id_type, typename M::vector_type>;
template<typename M, typename Attr> using constraint= typedef parallel::concurrent_unordered_map<typename M::id_type, Attr>;

}//namespace simpla{namespace manifold{


#endif //SIMPLA_CONSTRAINT_H
