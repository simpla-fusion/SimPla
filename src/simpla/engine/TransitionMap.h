//
// Created by salmon on 16-10-5.
//

#ifndef SIMPLA_TRANSITIONMAP_H
#define SIMPLA_TRANSITIONMAP_H

#include "simpla/algebra/nTuple.h"
#include "simpla/utilities/FancyStream.h"
#include "simpla/utilities/Log.h"
#include <type_traits>
#include "MeshBlock.h"

namespace simpla {
namespace engine {
class DomainBase;
class MeshAttributeVisitorBase;

template <typename T>
class MeshAttributeVisitor;

template <typename...>
struct TransitionMapView;

/**
 *   TransitionMapView: \f$\psi\f$,
 *   *Mapping: Two overlapped charts \f$x\in M\f$ and \f$y\in N\f$, and a mapping
 *    \f[
 *       \psi:M\rightarrow N,\quad y=\psi\left(x\right)
 *    \f].
 *   * Pull back: Let \f$g:N\rightarrow\mathbb{R}\f$ is a function on \f$N\f$,
 *     _pull-back_ of function \f$g\left(y\right)\f$ induce a function on \f$M\f$
 *     \f[
 *        \psi^{*}g\equiv g\circ\psi,\;\psi^{*}g=g\left(\psi\left(x\right)\right)
 *     \f]
 *
 *
 */
struct TransitionMap {
    TransitionMap(){};
    virtual ~TransitionMap(){};
    virtual id_type from_id() const = 0;
    virtual id_type to_id() const = 0;
    //    virtual point_type map(point_type const &x) const { return x; }
    //    point_type operator()(point_type const &x) const { return map(x); }
    //    virtual void push_forward(HeavyData const &src, HeavyData *dest) const =0;
    //    virtual void pull_back(HeavyData const &src, HeavyData *dest) const =0;

    virtual void Apply(DomainBase const &from, DomainBase &to) const = 0;
};
//
//template <typename M, typename N>
//struct TransitionMapView<M, N> : public TransitionMap {
//    typedef M l_mesh_type;
//    typedef M r_mesh_type;
//
//    std::shared_ptr<r_mesh_type> m_dst_;
//    std::shared_ptr<l_mesh_type> m_src_;
//    RectMesh m_overlap_;
//    Range<EntityId> m_range0_;
//
//    virtual id_type from_id() const { return m_src_->id(); };
//
//    virtual id_type to_id() const { return m_dst_->id(); };
//
//    TransitionMapView(std::shared_ptr<l_mesh_type> const &left, std::shared_ptr<r_mesh_type> const &right)
//        : m_src_(left), m_dst_(right) {}
//};

template <typename TM>
std::shared_ptr<TransitionMap> createTransitionMapView(std::shared_ptr<TM> const &src,
                                                       std::shared_ptr<TM> const &dest) {}
}
}  // namespace simpla { namespace mesh_as

#endif  // SIMPLA_TRANSITIONMAP_H
