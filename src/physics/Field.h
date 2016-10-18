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
        _apply(_impl::_assign(), other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator=(Other const &other)
    {
        _apply(_impl::_assign(), other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator+=(Other const &other)
    {
        _apply(_impl::plus_assign(), other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator-=(Other const &other)
    {
        _apply(_impl::minus_assign(), other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator*=(Other const &other)
    {
        _apply(_impl::multiplies_assign(), other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator/=(Other const &other)
    {
        _apply(_impl::divides_assign(), other);
        return *this;
    }

    /* @}*/
    template<typename ...Args> void
    assign(Args &&... args) { _apply(_impl::_assign(), std::forward<Args>(args)...); }

    template<typename ...Args> void
    apply(Args &&... args) { _apply(std::forward<Args>(args)...); }

private:


    template<typename TOP, typename ...Others> void
    _apply(TOP const &op, value_type const &v, Others &&...others)
    {
        base_type::apply(op, v, std::forward<Others>(others)...);
    }

    template<typename TOP, typename ...Others> void
    _apply(TOP const &op, this_type const &fexpr, Others &&...others)
    {
        base_type::apply(op,
                         [&](mesh::MeshEntityId const &s) -> value_type { return fexpr[s]; },
                         std::forward<Others>(others)...);
    }

    template<typename TOP, typename ...U, typename ...Others> void
    _apply(TOP const &op, Field<U...> const &fexpr, Others &&...others)
    {
        base_type::apply(op,
                         [&](mesh::MeshEntityId const &s) -> value_type { return this->mesh()->eval(fexpr, s); },
                         std::forward<Others>(others)...);
    }

    template<typename TOP, typename TFun, typename ...Others> void
    _apply(TOP const &op, TFun const &fun, Others &&...others)
    {
        base_type::apply(op,
                         [&](mesh::MeshEntityId const &s) { return fun(s); },
                         std::forward<Others>(others)...);
    }

public:

    template<typename TOP, typename TFun, typename ...Others> void
    apply_function(TOP const &op, TFun const &fexpr, Others &&...others)
    {
        base_type::apply(op,
                         [&](mesh::MeshEntityId const &s) -> value_type
                         {
                             return this->mesh()->template sample<IFORM>(s, fexpr(s));
                         },
                         std::forward<Others>(others)...);
    }

};

}//namespace simpla







#endif //SIMPLA_FIELD_H
