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

#include <simpla/mesh/Attribute.h>
#include <simpla/mesh/Worker.h>

#include "FieldTraits.h"
#include "FieldExpression.h"

namespace simpla
{

template<typename ...> class Field;

template<typename TV, typename TManifold, mesh::MeshEntityType IFORM> using field_t= Field<TV, TManifold, index_const<IFORM>>;


template<typename TV, typename TManifold, size_t I>
class Field<TV, TManifold, index_const<I >>
        : public mesh::AttributeView<TV, TManifold, static_cast<mesh::MeshEntityType>(I)>
{
private:
    static_assert(std::is_base_of<mesh::MeshBlock, TManifold>::value, "TManifold is not derived from MeshBlock");

    typedef Field<TV, TManifold, index_const<I >> this_type;

    typedef mesh::AttributeView<TV, TManifold, static_cast<mesh::MeshEntityType>(I)> base_type;

public:
    typedef typename traits::field_value_type<this_type>::type field_value_type;
    using typename base_type::mesh_type;
    using typename base_type::value_type;

private:
    using base_type::m_mesh_;
    using base_type::m_data_;
public:

    Field() : base_type() {};

//    template<typename ...Args>
//    Field(Args &&...args)  : base_type(std::forward<Args>(args)...) {};

    template<typename ...Args>
    Field(mesh_type *m = nullptr, std::string const &s = "", Args &&...args) :
            base_type(m, s, std::forward<Args>(args)...) {};

    template<typename ...Args>
    Field(std::shared_ptr<mesh_type> const &m, std::string const &s = "", Args &&...args) :
            base_type(m, s, std::forward<Args>(args)...) {};

    template<typename ...Args>
    Field(std::string const &s, Args &&...args) :
            base_type(nullptr, s, std::forward<Args>(args)...) {};

    template<typename ...Args>
    Field(std::shared_ptr<mesh::Attribute> const &attr, Args &&...args) :
            base_type(attr, std::forward<Args>(args)...) {};

    virtual ~Field() {}

    Field(this_type const &other) = delete;

    Field(this_type &&other) = delete;

    virtual bool is_a(std::type_info const &t_info) const
    {
        return t_info == typeid(this_type) || base_type::is_a(t_info);
    };


    /** @name as_function  @{*/
    template<typename ...Args> field_value_type
    gather(Args &&...args) const { return m_mesh_->gather(*this, std::forward<Args>(args)...); }

    template<typename ...Args> field_value_type
    operator()(Args &&...args) const { return gather(std::forward<Args>(args)...); }
    /**@}*/

    /** @name as_array   @{*/

    this_type &operator=(this_type const &other)
    {
        apply_dispatch(_impl::_assign(), mesh::SP_ES_ALL, other);
        return *this;
    }

    template<typename ...U> inline this_type &
    operator=(Field<U...> const &other)
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
        base_type::apply(op, r0, static_cast< typename base_type::expression_tag *>(nullptr), expr);
    }

    template<typename TOP, typename TRange, typename ...U> void
    apply_dispatch(TOP const &op, TRange r0, std::function<value_type(point_type const &, U const &...)> const &fun,
                   U &&...args)
    {
        base_type::apply(op, r0, static_cast<typename base_type::function_tag *>(nullptr), fun, std::forward<U>(args)...);
    }

    template<typename TOP, typename TRange, typename Other> void
    apply_dispatch(TOP const &op, TRange r0, Other const &other)
    {
        base_type::apply(op, r0, static_cast<  typename base_type::scalar_value_tag *>(nullptr), other);
    }






//
//    template<typename TOP, typename TFun, typename ...Args> void
//    apply(TOP const &op, mesh::EntityIdRange const r0, field_function_tag const *, TFun const &fun, Args &&...args)
//    {
//        deploy();
//        r0.foreach(
//                [&](mesh::MeshEntityId const &s)
//                {
//                    op(get(s), m_mesh_->template sample<IFORM>(s, fun(m_mesh_->point(s), std::forward<Args>(args)...)));
//                });
//    }
//    template<typename TOP, typename TFun> void
//    apply_function_with_define_domain(TOP const &op, mesh::EntityIdRange const r0,
//                                      std::function<Real(point_type const &)> const &geo,
//                                      TFun const &fun)
//    {
//        deploy();
//        r0.foreach([&](mesh::MeshEntityId const &s)
//                   {
//                       auto x = m_mesh_->point(s);
//                       if (geo(x) < 0)
//                       {
//                           op(m_data_->get(s), m_mesh_->template sample<IFORM>(s, fun(x)));
//                       }
//                   });
//    }

    void copy(mesh::EntityIdRange const &r0, this_type const &g)
    {
//        r0.foreach([&](mesh::MeshEntityId const &s) { base_type::get(s) = g.base_type::get(s); });
    }


    virtual void copy(mesh::EntityIdRange const &r0, mesh::DataBlock const &other)
    {
//        assert(other.is_a(typeid(this_type)));
//
//        this_type const &g = static_cast<this_type const & >(other);
//
//        copy(r0, static_cast<this_type const & >(other));

    }


};

namespace traits
{
template<typename TV, typename TM, size_t I>
struct reference<Field<TV, TM, index_const<I> > > { typedef Field<TV, TM, index_const<I> > const &type; };
}
}//namespace simpla







#endif //SIMPLA_FIELD_H
