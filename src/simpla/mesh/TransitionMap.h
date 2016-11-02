//
// Created by salmon on 16-10-5.
//

#ifndef SIMPLA_TRANSITIONMAP_H
#define SIMPLA_TRANSITIONMAP_H

#include <type_traits>
#include <simpla/toolbox/Log.h>
#include <simpla/toolbox/nTuple.h>
#include <simpla/toolbox/nTupleExt.h>
#include <simpla/toolbox/PrettyStream.h>
#include "MeshCommon.h"
#include "MeshBlock.h"

namespace simpla { namespace mesh
{


class MeshAttributeVisitorBase;

template<typename T> class MeshAttributeVisitor;

template<typename ...> struct TransitionMap;

/**
 *   TransitionMap: \f$\psi\f$,
 *   *Mapping: Two overlapped charts \f$x\in M\f$ and \f$y\in N\f$, and a mapping
 *    \f[
 *       \psi:M\rightarrow N,\quad y=\psi\left(x\right)
 *    \f].
 *   * Pull back: Let \f$g:N\rightarrow\mathbb{R}\f$ is a function on \f$N\f$,
 *     _pull-back_ of function \f$g\left(y\right)\f$ induce a function on \f$M\f$
 *   \f[
 *       \psi^{*}g&\equiv g\circ\psi,\;\psi^{*}g=&g\left(\psi\left(x\right)\right)
 *   \f]
 *
 *
 */
struct TransitionMapBase : public toolbox::Object
{

    TransitionMapBase() {};

    virtual  ~TransitionMapBase() {};

    virtual MeshBlock::id_type from_id() const =0;

    virtual MeshBlock::id_type to_id() const =0;
//    virtual point_type map(point_type const &x) const { return x; }
//    point_type operator()(point_type const &x) const { return map(x); }
//    virtual void push_forward(DataEntityHeavy const &src, DataEntityHeavy *dest) const =0;
//    virtual void pull_back(DataEntityHeavy const &src, DataEntityHeavy *dest) const =0;


};


template<typename M, typename N>
struct TransitionMap<M, N> : public TransitionMapBase
{
    typedef M l_mesh_type;
    typedef M r_mesh_type;

    std::shared_ptr<r_mesh_type> m_dst_;
    std::shared_ptr<l_mesh_type> m_src_;
    MeshBlock m_overlap_;
    EntityRange m_range0_;

    virtual MeshBlock::id_type from_id() const { return m_src_->id(); };

    virtual MeshBlock::id_type to_id() const { return m_dst_->id(); };

    TransitionMap(std::shared_ptr<l_mesh_type> const &left, std::shared_ptr<r_mesh_type> const &right)
            : m_src_(left), m_dst_(right) {}


};


template<typename TM>
std::shared_ptr<TransitionMapBase>
createTransitionMap(std::shared_ptr<TM> const &src, std::shared_ptr<TM> const &dest)
{

}
}}//namespace simpla { namespace mesh_as

#endif //SIMPLA_TRANSITIONMAP_H
