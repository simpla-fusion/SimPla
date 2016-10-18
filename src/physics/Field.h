/**
 * @file field.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_H
#define SIMPLA_FIELD_H

#include "SIMPLA_config.h"

#include <type_traits>
#include <cassert>

#include "../toolbox/type_traits.h"
#include "../toolbox/DataSet.h"
#include "../mesh/MeshCommon.h"
#include "../mesh/MeshBase.h"
#include "../mesh/ModelSelect.h"
#include "../mesh/Attribute.h"

#include "FieldTraits.h"
#include "FieldExpression.h"

namespace simpla
{

template<typename ...> class Field;

template<typename TV, typename TManifold, size_t IFORM> using field_t= Field<TV, TManifold, index_const<IFORM>>;


template<typename TV, typename TManifold, size_t IFORM>
class Field<TV, TManifold, index_const<IFORM>>
        : public mesh::Attribute<TV, TManifold, static_cast<mesh::MeshEntityType>(IFORM)>
{
private:
    static_assert(std::is_base_of<mesh::MeshBase, TManifold>::value, "TManifold is not derived from MeshBase");

    typedef Field<TV, TManifold, index_const<IFORM>> this_type;

    typedef mesh::Attribute<TV, TManifold, static_cast<mesh::MeshEntityType>(IFORM)> base_type;

public:
    using base_type::iform;

    using base_type::deploy;

    using typename base_type::mesh_type;

    using typename base_type::value_type;

    typedef typename traits::field_value_type<this_type>::type field_value_type;


    //create construct
    Field(mesh_type const *m = nullptr, std::shared_ptr<value_type> p = nullptr) : base_type(m, p) {};

    Field(std::shared_ptr<mesh_type const> m, std::shared_ptr<value_type> p = nullptr) : base_type(m.get(), p) {};

    //copy construct
    Field(this_type const &other) : base_type(other) {}

    ~Field() {}

    /** @name as_function  @{*/

    template<typename ...Args> field_value_type
    gather(Args &&...args) const { return base_type::m_mesh_->gather(*this, std::forward<Args>(args)...); }


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
        r0.foreach([&](mesh::MeshEntityId const &s) { op(this->get(s), v); });
    }

    template<typename TOP> void
    apply(TOP const &op, mesh::EntityRange const r0, this_type const &other)
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { op(this->get(s), other.get(s)); });

    }

    template<typename TOP, typename TFun> void
    apply(TOP const &op, mesh::EntityRange const r0, TFun const &fun,
          CHECK_FUNCTION_SIGNATURE(value_type, TFun(mesh::MeshEntityId const &))
    )
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { op(this->get(s), fun(s)); });

    }

    template<typename TOP, typename ...U> void
    apply(TOP const &op, mesh::EntityRange const r0, Field<U...> const &fexpr)
    {
        deploy();
        mesh_type const &m = *this->mesh();
        r0.foreach([&](mesh::MeshEntityId const &s) { op(this->get(s), m.eval(fexpr, s)); });
    }

    template<typename TOP, typename Arg> void
    apply(TOP const &op, Arg const &v)
    {
        apply(op, this->mesh()->range(iform, mesh::SP_ES_ALL), v);
    }

    template<typename ...Args> void
    assign(Args &&... args)
    {
        apply(_impl::_assign(), std::forward<Args>(args)...);
    }


    template<typename TOP, typename TFun, typename ...Args> void
    apply_function(TOP const &op, mesh::EntityRange const r0, TFun const &fun, Args &&...args)
    {
        deploy();
        mesh_type const &m = *this->mesh();
        r0.foreach(
                [&](mesh::MeshEntityId const &s)
                {
                    op(this->get(s), m.template sample<IFORM>(s, fun(m.point(s), std::forward<Args>(args)...)));
                });
    }

    template<typename TOP, typename TFun> void
    apply_function_in_geometric_domain(TOP const &op, mesh::EntityRange const r0,
                                       std::function<Real(point_type const &)> const &geo,
                                       TFun const &fun)
    {
        deploy();
        mesh_type const &m = *this->mesh();
        r0.foreach([&](mesh::MeshEntityId const &s)
                   {
                       auto x = m.point(s);
                       if (geo(x) < 0)
                       {
                           op(this->get(s), m.template sample<IFORM>(s, fun(x)));
                       }
                   });
    }


    template<typename ...Args> void
    assign_function(Args &&... args)
    {
        apply_function(_impl::_assign(), this->mesh()->range(iform), std::forward<Args>(args)...);
    }

};

}//namespace simpla







#endif //SIMPLA_FIELD_H
