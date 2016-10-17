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
        base_type::apply(_impl::_assign(), other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator=(Other const &other)
    {
        base_type::apply(_impl::_assign(), other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator+=(Other const &other)
    {
        base_type::apply(_impl::plus_assign(), other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator-=(Other const &other)
    {
        base_type::apply(_impl::minus_assign(), other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator*=(Other const &other)
    {
        base_type::apply(_impl::multiplies_assign(), other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator/=(Other const &other)
    {
        base_type::apply(_impl::divides_assign(), other);
        return *this;
    }

    /* @}*/

    void
    assign(mesh::EntityRange const &r0, value_type const &v)
    {
        base_type::apply2(r0, _impl::_assign(), v);
    }

    template<typename TFun> void
    assign(mesh::EntityRange const &r0, TFun const &op,
           CHECK_FUNCTION_SIGNATURE(field_value_type, TFun(point_type const&, field_value_type const &)))
    {
        auto const &m = *this->mesh();
        base_type::apply2(
                r0, _impl::_assign(),
                [&](mesh::MeshEntityId const &s)
                {
                    auto x = m.point(s);
                    return m.template sample<IFORM>(s, op(x, this->gather(x)));
                }
        );
    }

    template<typename TFun> void
    assign(mesh::EntityRange const &r0, TFun const &op,
           CHECK_FUNCTION_SIGNATURE(field_value_type, TFun(point_type const&)))
    {
        auto const &m = *this->mesh();

        base_type::apply2(
                r0, _impl::_assign(),
                [&](mesh::MeshEntityId const &s)
                {
                    return m.template sample<IFORM>(s, op(m.point(s)));
                });
    }

};
}//namespace simpla

//public:
//
//
//    template<typename TOP>
//    this_type &
//    apply(TOP const &op)
//    {
//        deploy();
//
////        apply(m_mesh_->range(iform, mesh::SP_ES_NON_LOCAL), op);
////        base_type::nonblocking_sync();
////        apply(m_mesh_->range(iform, mesh::SP_ES_LOCAL), op);
////        base_type::wait();
////        apply(m_mesh_->range(iform, mesh::SP_ES_VALID), op);
//
//        return *this;
//    }
//
//    template<typename Other>
//    this_type &
//    fill(Other const &other)
//    {
//        this->deploy();
//
////        entity_id_range(mesh::SP_ES_ALL).foreach([&](mesh::MeshEntityId const &s) { get(s) = other; });
//
//        return *this;
//    }






#endif //SIMPLA_FIELD_H
