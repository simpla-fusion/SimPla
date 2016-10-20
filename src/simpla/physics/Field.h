/**
 * @file field.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_H
#define SIMPLA_FIELD_H

#include <simpla/SIMPLA_config.h>

#include <type_traits>
#include <cassert>

#include <simpla/toolbox/type_traits.h>
#include <simpla/toolbox/DataSet.h>

#include <simpla/mesh/MeshCommon.h>
#include <simpla/mesh/MeshBase.h>
#include <simpla/mesh/Attribute.h>
#include <simpla/mesh/Patch.h>

#include "FieldTraits.h"
#include "FieldExpression.h"

namespace simpla
{

template<typename ...> class Field;

template<typename TV, typename TManifold, size_t IFORM> using field_t= Field<TV, TManifold, index_const<IFORM>>;


template<typename TV, typename TManifold, size_t IFORM>
class Field<TV, TManifold, index_const<IFORM>>
        : public mesh::Attribute<TV, TManifold, static_cast< mesh::MeshEntityType>(IFORM)>
{
private:
    static_assert(std::is_base_of<mesh::MeshBase, TManifold>::value, "TManifold is not derived from MeshBase");

    typedef Field<TV, TManifold, index_const<IFORM>> this_type;

    typedef mesh::Attribute<TV, TManifold, static_cast<mesh::MeshEntityType>(IFORM)> base_type;


public:
    using base_type::iform;

    using typename base_type::mesh_type;

    using typename base_type::value_type;

    typedef typename traits::field_value_type<this_type>::type field_value_type;


public:
    //create construct
//    Field(mesh_type const *m = nullptr, std::shared_ptr<value_type> p = nullptr) : base_type(m, p) {};

    Field(std::shared_ptr<mesh_type> m) : base_type(m) {};

    //copy construct
    Field(this_type &&other) : base_type(std::move(other)) {};

    ~Field() {}

    using base_type::m_mesh_;
    using base_type::m_patch_;
    using base_type::deploy;


    /** @name as_function  @{*/

    template<typename ...Args> field_value_type
    gather(Args &&...args) const { return m_mesh_->gather(*this, std::forward<Args>(args)...); }


    template<typename ...Args> field_value_type
    operator()(Args &&...args) const { return gather(std::forward<Args>(args)...); }
    /**@}*/

    /** @name as_array   @{*/
    this_type &operator=(this_type const &other)
    {
        apply(_impl::_assign(), other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator=(Other const &other)
    {
        apply(_impl::_assign(), other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator+=(Other const &other)
    {
        apply(_impl::plus_assign(), other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator-=(Other const &other)
    {
        apply(_impl::minus_assign(), other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator*=(Other const &other)
    {
        apply(_impl::multiplies_assign(), other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator/=(Other const &other)
    {
        apply(_impl::divides_assign(), other);
        return *this;
    }

    /* @}*/


    template<typename TOP> void
    apply(TOP const &op, mesh::EntityRange const r0, value_type const &v)
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { op(m_patch_->get(s), v); });
    }

    template<typename TOP> void
    apply(TOP const &op, mesh::EntityRange const r0, this_type const &other)
    {
        deploy();
        r0.foreach([&](
                mesh::MeshEntityId const &s) { op(m_patch_->get(s), other.m_patch_->get(s)); });

    }

    template<typename TOP, typename ...U> void
    apply(TOP const &op, mesh::EntityRange const r0, Field<U...> const &fexpr)
    {
        deploy();
        r0.foreach([&](
                mesh::MeshEntityId const &s) { op(m_patch_->get(s), m_mesh_->eval(fexpr, s)); });
    }

    template<typename TOP, typename TFun> void
    apply(TOP const &op, mesh::EntityRange const r0, TFun const &fun)
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { op(m_patch_->get(s), fun(s)); });
    }

    template<typename TOP, typename Arg> void
    apply(TOP const &op, Arg const &v)
    {
        deploy();
        apply(op, m_mesh_->range(iform, mesh::SP_ES_ALL), v);
    }

    template<typename TOP, typename Arg> void
    apply(TOP const &op, mesh::MeshZoneTag tag, Arg const &v)
    {
        deploy();
        apply(op, m_mesh_->range(iform, tag), v);
    }

    template<typename ...Args> void
    assign(Args &&... args)
    {
        deploy();
        apply(_impl::_assign(), std::forward<Args>(args)...);
    }


    template<typename TOP, typename TFun, typename ...Args> void
    apply_function(TOP const &op, mesh::EntityRange const r0, TFun const &fun, Args &&...args)
    {
        deploy();
        r0.foreach(
                [&](mesh::MeshEntityId const &s)
                {
                    op(m_patch_->get(s),
                       m_mesh_->
                               template sample<IFORM>(s, fun(m_mesh_->point(s), std::forward<Args>(args)...)));
                });
    }

    template<typename ...Args> void
    assign_function(Args &&... args)
    {
        apply_function(_impl::_assign(), m_mesh_->range(iform), std::forward<Args>(args)...);
    }

    template<typename TOP, typename TFun> void
    apply_function_with_define_domain(TOP const &op, mesh::EntityRange const r0,
                                      std::function<Real(point_type const &)> const &geo,
                                      TFun const &fun)
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s)
                   {
                       auto x = m_mesh_->point(s);
                       if (geo(x) < 0)
                       {
                           op(m_patch_->get(s), m_mesh_->template sample<IFORM>(s, fun(x)));
                       }
                   });
    }
};

}//namespace simpla







#endif //SIMPLA_FIELD_H
