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
        : public mesh::Attribute<mesh::Patch<TV, TManifold, static_cast<mesh::MeshEntityType >(IFORM) >>
{
private:
    static_assert(std::is_base_of<mesh::MeshBase, TManifold>::value, "TManifold is not derived from MeshBase");

    typedef Field<TV, TManifold, index_const<IFORM>> this_type;

    typedef mesh::Attribute<mesh::Patch<TV, TManifold, static_cast<mesh::MeshEntityType >(IFORM)  >> base_type;


public:

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


    using base_type::get;
    using base_type::operator[];
    using base_type::deploy;
    using base_type::apply;

    /** @name as_function  @{*/

    template<typename ...Args> field_value_type
    gather(Args &&...args) const { return base_type::m_patch_->mesh()->gather(*this, std::forward<Args>(args)...); }

    template<typename ...Args> field_value_type
    operator()(Args &&...args) const { return gather(std::forward<Args>(args)...); }
    /**@}*/

    /** @name as_array   @{*/

    this_type &operator=(this_type const &other)
    {
        apply_dispatch(_impl::_assign(), mesh::SP_ES_ALL, other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator=(Other const &other)
    {
        apply_dispatch(_impl::_assign(), mesh::SP_ES_ALL, other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator+=(Other const &other)
    {
        apply_dispatch(_impl::plus_assign(), mesh::SP_ES_ALL, other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator-=(Other const &other)
    {
        apply_dispatch(_impl::minus_assign(), mesh::SP_ES_ALL, other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator*=(Other const &other)
    {
        apply_dispatch(_impl::multiplies_assign(), mesh::SP_ES_ALL, other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator/=(Other const &other)
    {
        apply_dispatch(_impl::divides_assign(), mesh::SP_ES_ALL, other);
        return *this;
    }

    template<typename TOther> void
    assign(TOther const &v)
    {
        apply_dispatch(_impl::_assign(), mesh::SP_ES_ALL, v);
    }

    template<typename TRange, typename ...Args> void
    assign(TRange const &r0, Args &&...args)
    {
        apply_dispatch(_impl::_assign(), r0, std::forward<Args>(args)...);
    }
    /* @}*/
private:

    template<typename TOP, typename TRange, typename ...U>
    void apply_dispatch(TOP const &op, TRange r0, Field<Expression<U...>> const &expr)
    {
        base_type::apply(op, r0, static_cast<typename base_type::patch_type::expression_tag *>(nullptr), expr);
    }

    template<typename TOP, typename TRange, typename ...U> void
    apply_dispatch(TOP const &op, TRange r0, std::function<value_type(point_type const &, U const &...)> const &fun,
                   U &&...args)
    {
        base_type::apply(op, r0,
                         static_cast<typename base_type::patch_type::function_tag *>(nullptr),
                         fun, std::forward<U>(args)...);
    }

    template<typename TOP, typename TRange, typename Other> void
    apply_dispatch(TOP const &op, TRange r0, Other const &other)
    {
        base_type::apply(op, r0, other);
    }


public:
    template<typename TOP, typename TRange, typename ...Args> void
    apply_function(TOP const &op, TRange r0, Args &&...args)
    {
        base_type::apply(op, r0, static_cast<typename base_type::patch_type::function_tag *>(nullptr),
                         std::forward<Args>(args)...);
    }

    template<typename ...Args> void
    assign_function(Args &&...args) { apply_function(_impl::_assign(), std::forward<Args>(args)...); }
};

}//namespace simpla







#endif //SIMPLA_FIELD_H
