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

template<typename TV, typename TManifold, size_t IFORM> using field_t= Field<TV, TManifold, index_const<IFORM>>;


template<typename TV, typename TManifold, size_t IFORM>
class Field<TV, TManifold, index_const<IFORM>> : public mesh::AttributeView
{
private:
    static_assert(std::is_base_of<mesh::MeshBlock, TManifold>::value, "TManifold is not derived from MeshBlock");

    typedef Field<TV, TManifold, index_const<IFORM>> this_type;

    typedef mesh::AttributeView base_type;

    static constexpr mesh::MeshEntityType iform = static_cast<mesh::MeshEntityType>(IFORM);

    typedef TManifold mesh_type;

    typedef TV value_type;

    typedef typename traits::field_value_type<this_type>::type field_value_type;

private:
    typedef typename TManifold::template data_block_type<TV, iform> data_block_type;
    static constexpr int ndims = mesh_type::ndims;
    mesh_type const *m_mesh_ = nullptr;
    data_block_type *m_data_ = nullptr;

public:
    template<typename ...Args>
    Field(Args &&...args) : base_type(std::forward<Args>(args)...) {};

    virtual ~Field() {}

    virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(this_type); };

    virtual void
    create(mesh::MeshBlock const *m, bool is_scratch = false)
    {
        auto res = std::dynamic_pointer_cast<mesh::DataBlock>(std::make_shared<data_block_type>(m));
        if (!is_scratch) { base_type::attribute()->insert(m, res); };
    };

    void deploy(mesh::MeshBlock const *m = nullptr)
    {
        base_type::deploy(m);
        m_mesh_ = base_type::template mesh<mesh_type>();
        m_data_ = base_type::template data<data_block_type>();
    }

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
        apply(op, r0, static_cast< expression_tag *>(nullptr), expr);
    }

    template<typename TOP, typename TRange, typename ...U> void
    apply_dispatch(TOP const &op, TRange r0, std::function<value_type(point_type const &, U const &...)> const &fun,
                   U &&...args)
    {
        apply(op, r0, static_cast< function_tag *>(nullptr), fun, std::forward<U>(args)...);
    }

    template<typename TOP, typename TRange, typename Other> void
    apply_dispatch(TOP const &op, TRange r0, Other const &other)
    {
        apply(op, r0, other);
    }


public:
    template<typename TOP, typename TRange, typename ...Args> void
    apply_function(TOP const &op, TRange r0, Args &&...args)
    {
        apply(op, r0, static_cast< function_tag *>(nullptr), std::forward<Args>(args)...);
    }

    template<typename ...Args> void
    assign_function(Args &&...args) { apply_function(_impl::_assign(), std::forward<Args>(args)...); }


    virtual void clear() { m_data_->clear(); };

    inline value_type &get(mesh::MeshEntityId const &s) { return m_data_->get(s.x, s.y, s.z, s.w); }

    inline value_type const &get(mesh::MeshEntityId const &s) const { return m_data_->get(s.x, s.y, s.z, s.w); }

    inline value_type &operator[](mesh::MeshEntityId const &s) { return m_data_->get(s.x, s.y, s.z, s.w); }

    inline value_type const &operator[](mesh::MeshEntityId const &s) const
    {
        return m_data_->get(s.x, s.y, s.z, s.w);
    }

    struct expression_tag {};
    struct function_tag {};
    struct field_function_tag {};

    template<typename TOP, typename ...Args> void
    apply(TOP const &op, mesh::MeshZoneTag tag, Args &&...args)
    {
        deploy();
        apply(op, m_mesh_->range(IFORM, tag), std::forward<Args>(args)...);
    }

    template<typename TOP, typename TRange> void
    apply(TOP const &op, TRange const &r0, value_type const &v)
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { op(get(s), v); });
    }

    template<typename TOP> void
    apply(TOP const &op, mesh::EntityIdRange const &r0, this_type const &other)
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { op(get(s), other.get(s)); });
    }


    template<typename TOP, typename TFun> void
    apply(TOP const &op, mesh::EntityIdRange const &r0, TFun const &fun)
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { op(get(s), fun(s)); });
    }


    template<typename TOP, typename TFun, typename ...Args> void
    apply(TOP const &op, mesh::EntityIdRange const r0, function_tag const *, TFun const &fun, Args &&...args)
    {
        deploy();
        r0.foreach(
                [&](mesh::MeshEntityId const &s)
                {
                    op(get(s), fun(m_mesh_->point(s), std::forward<Args>(args)...));
                });
    }

    template<typename TOP, typename ...TExpr> void
    apply(TOP const &op, mesh::EntityIdRange const &r0, expression_tag const *, TExpr &&...fexpr)
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s)
                   {
                       op(get(s), m_mesh_->eval(std::forward<TExpr>(fexpr), s)...);
                   });
    }

    template<typename TOP, typename TFun, typename ...Args> void
    apply(TOP const &op, mesh::EntityIdRange const r0, field_function_tag const *, TFun const &fun, Args &&...args)
    {
        deploy();
        r0.foreach(
                [&](mesh::MeshEntityId const &s)
                {
                    op(get(s), m_mesh_->template sample<IFORM>(s, fun(m_mesh_->point(s), std::forward<Args>(args)...)));
                });
    }



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
        r0.foreach([&](mesh::MeshEntityId const &s) { get(s) = g.get(s); });
    }


    virtual void copy(mesh::EntityIdRange const &r0, mesh::DataBlock const &other)
    {
//        assert(other.is_a(typeid(this_type)));
//
//        this_type const &g = static_cast<this_type const & >(other);
//
//        copy(r0, static_cast<this_type const & >(other));

    }


private:

};
}//namespace simpla







#endif //SIMPLA_FIELD_H
