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
#include <simpla/toolbox/Log.h>
#include <simpla/toolbox/design_pattern/Observer.h>
#include <simpla/mesh/Attribute.h>
#include <simpla/mesh/Worker.h>
#include <simpla/mesh/MeshCommon.h>

#include "FieldTraits.h"
#include "FieldExpression.h"
#include "schemes/CalculusPolicy.h"
#include "schemes/InterpolatePolicy.h"


namespace simpla
{


template<typename ...> class Field;


template<typename TV, typename TM, size_type I, size_type DOF>
class Field<TV, TM, index_const<I>, index_const<DOF>> :
        public mesh::AttributeView<TV, static_cast<mesh::MeshEntityType >(I), DOF>
{
private:
    static constexpr mesh::MeshEntityType IFORM = static_cast<mesh::MeshEntityType>(I);

    typedef Field<TV, TM, index_const<I>, index_const<DOF>> this_type;
    typedef mesh::AttributeView<TV, IFORM, DOF> base_type;

public:
    typedef TV value_type;
    typedef TM mesh_type;
    typedef typename std::conditional<DOF == 1, value_type, nTuple<value_type, DOF> >::type cell_tuple;
    typedef typename std::conditional<(IFORM == mesh::VERTEX || IFORM == mesh::VOLUME),
            cell_tuple, nTuple<cell_tuple, 3> >::type field_value_type;

private:
    typedef typename mesh_type::template data_block_type<TV, IFORM, DOF> data_block;

    data_block *m_data_;
    mesh_type const *m_;

public:

    typedef manifold::schemes::CalculusPolicy<mesh_type> calculus_policy;

    typedef manifold::schemes::InterpolatePolicy<mesh_type> interpolate_policy;


    template<typename ...Args>
    explicit Field(mesh_type *m, Args &&...args):base_type(m, std::forward<Args>(args)...), m_(m), m_data_(nullptr)
    {
    };

    virtual ~Field() {}

    Field(this_type const &other) = delete;

    Field(this_type &&other) = delete;

    virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(this_type); };

    bool is_valid() const { return m_data_ != nullptr && m_ != nullptr; };

    using base_type::entity_type;

    using base_type::value_type_info;

    using base_type::dof;

    void deploy()
    {

        m_data_ = static_cast<data_block *>(base_type::data());
        m_data_->deploy();
    }


    virtual void clear()
    {
        deploy();
        m_data_->clear();
    }


    /** @name as_function  @{*/
    template<typename ...Args> field_value_type
    gather(Args &&...args) const { return m_->gather(*this, std::forward<Args>(args)...); }

    template<typename ...Args> field_value_type
    operator()(Args &&...args) const { return gather(std::forward<Args>(args)...); }

    /**@}*/

    /** @name as_array   @{*/

    template<typename ...Args>
    inline value_type &get(Args &&...args) { return m_data_->get(std::forward<Args>(args)...); }

    template<typename ...Args>
    inline value_type const &get(Args &&...args) const { return m_data_->get(std::forward<Args>(args)...); }

    template<typename ...Args>
    inline value_type &operator()(Args &&...args) { return m_data_->get(std::forward<Args>(args)...); }

    template<typename ...Args>
    inline value_type const &operator()(Args &&...args) const { return m_data_->get(std::forward<Args>(args)...); }

    template<typename TI>
    inline value_type &operator[](TI const &s) { return m_data_->get(s); }

    template<typename TI>
    inline value_type const &operator[](TI const &s) const { return m_data_->get(s); }

    this_type &operator=(this_type const &other)
    {
        foreach(mesh::SP_ES_ALL, other);
        return *this;
    }

    template<typename ...U> inline this_type &
    operator=(Field<U...> const &other)
    {
        foreach(mesh::SP_ES_ALL, other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator=(Other const &other)
    {
        foreach(mesh::SP_ES_ALL, other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator+=(Other const &other)
    {
        *this = *this + other;
        return *this;
    }

    template<typename Other> inline this_type &
    operator-=(Other const &other)
    {
        *this = *this - other;
        return *this;
    }

    template<typename Other> inline this_type &
    operator*=(Other const &other)
    {
        *this = *this * other;
        return *this;
    }

    template<typename Other> inline this_type &
    operator/=(Other const &other)
    {
        *this = *this / other;
        return *this;
    }

    template<typename ...Args> void
    assign(Args &&...args) { foreach(std::forward<Args>(args)...); }

    /* @}*/

    template<typename U> void
    foreach(mesh::EntityIdRange const &r0, U const &v,
            ENABLE_IF((std::is_arithmetic<U>::value || std::is_same<U, value_type>::value)))
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { get(s) = v; });
    }


    template<typename ...U>
    void foreach(mesh::EntityIdRange const &r0, Field<Expression<U...>> const &expr)
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { get(s) = calculus_policy::eval(*m_, expr, s); });

    }

//    template<typename TFun, typename ...U> void
//    foreach(mesh::EntityIdRange const &r0, std::function<value_type(point_type const &, U const &...)> const &fun,
//            U &&...args)
//    {
//        deploy();
//        r0.foreach([&](mesh::MeshEntityId const &s) { get(s) = fun(m_->point(s), std::forward<U>(args)...); });
//    }


    template<typename TFun> void
    foreach(mesh::EntityIdRange const &r0, TFun const &fun,
            ENABLE_IF((std::is_same<typename std::result_of<TFun(point_type const &)>::type, field_value_type>::value))
    )
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s)
                   {
                       get(s) = interpolate_policy::template sample<IFORM>(*m_, s, fun(m_->point(s)));
                   });
    }

    template<typename TFun> void
    foreach(mesh::EntityIdRange const &r0, TFun const &fun,
            ENABLE_IF(
                    (std::is_same<typename std::result_of<TFun(mesh::MeshEntityId const &)>::type, value_type>::value))
    )
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { return fun(s); });
    }


    template<typename U> void
    foreach(U const &v,
            ENABLE_IF((std::is_arithmetic<U>::value || std::is_same<U, value_type>::value)))
    {
        deploy();
        m_data_->foreach([&](index_type i, index_type j, index_type k, index_type l) { return v; });
    }


    template<typename ...U>
    void foreach(Field<Expression<U...>> const &expr)
    {
        deploy();
        auto b = std::get<0>(m_->mesh_block()->outer_index_box());
        index_type gw[4] = {1, 1, 1, 0};

        m_data_->foreach(
                [&](index_type i, index_type j, index_type k, index_type l)
                {
                    auto s = mesh::MeshEntityIdCoder::pack_index4<IFORM>(i, j, k, l / DOF, l % DOF);
                    return calculus_policy::eval(*m_, expr, s);
                });

    }

    template<typename TFun> void
    foreach(TFun const &fun,
            ENABLE_IF((std::is_same<typename std::result_of<TFun(point_type const &)>::type, value_type>::value)))
    {
        deploy();

        m_data_->foreach(
                [&](index_type i, index_type j, index_type k, index_type l)
                {
                    return fun(m_->point(mesh::MeshEntityIdCoder::pack_index4<IFORM>(i, j, k, l / DOF, l % DOF)));
                });


    }

    template<typename TFun> void
    foreach(TFun const &fun,
            ENABLE_IF((std::is_same<typename std::result_of<TFun(index_type, index_type, index_type,
                                                                 index_type)>::type, value_type>::value)))
    {
        deploy();

        m_data_->foreach([&](index_type i, index_type j, index_type k, index_type l) { return fun(i, j, k, l); });


    }

    template<typename TFun> void
    foreach(TFun const &fun,
            ENABLE_IF(((!std::is_same<typename std::result_of<TFun(point_type const &)>::type, value_type>::value) &&
                       (std::is_same<typename std::result_of<TFun(point_type const &)>::type, field_value_type>::value)
                      ))
    )
    {
        deploy();

        m_data_->foreach(
                [&](index_type i, index_type j, index_type k, index_type l)
                {
                    auto s = mesh::MeshEntityIdCoder::pack_index4<IFORM>(i, j, k, l);
                    return interpolate_policy::template sample<IFORM>(*m_, s, fun(m_->point(s)));
                });


    }


    template<typename TFun> void
    foreach(TFun const &fun,
            ENABLE_IF(
                    (std::is_same<typename std::result_of<TFun(mesh::MeshEntityId const &)>::type, value_type>::value))
    )
    {
        deploy();
        m_data_->foreach(
                [&](index_type i, index_type j, index_type k, index_type l)
                {
                    return fun(mesh::MeshEntityIdCoder::pack_index4<IFORM>(i, j, k, l));
                });
    }

    template<typename U> void
    foreach_ghost(U const &v,
                  ENABLE_IF((std::is_arithmetic<U>::value || std::is_same<U, value_type>::value)))
    {
        deploy();

        m_data_->foreach_ghost([&](index_type i, index_type j, index_type k, index_type l) { return v; });
    }


    template<typename ...U>
    void foreach_ghost(Field<Expression<U...>> const &expr)
    {
        deploy();

        m_data_->foreach_ghost(
                [&](index_type i, index_type j, index_type k, index_type l)
                {
                    return calculus_policy::eval(*m_, expr, mesh::MeshEntityIdCoder::pack_index4<IFORM>(i, j, k, l));
                });

    }


    template<typename TFun> void
    foreach_ghost(TFun const &fun,
                  ENABLE_IF((std::is_same<typename std::result_of<TFun(point_type const &)>::type,
                          field_value_type>::value)))
    {
        deploy();
        m_data_->foreach_ghost(
                [&](index_type i, index_type j, index_type k, index_type l)
                {
                    auto s = mesh::MeshEntityIdCoder::pack_index4<IFORM>(i, j, k, l);
                    return interpolate_policy::template sample<IFORM>(*m_, s, fun(m_->point(s)));
                });


    }

    template<typename TFun> void
    foreach_ghost(TFun const &fun,
                  ENABLE_IF(
                          (std::is_same<typename std::result_of<TFun(
                                  mesh::MeshEntityId const &)>::type, value_type>::value))
    )
    {
        deploy();

        m_data_->foreach_ghost(
                [&](index_type i, index_type j, index_type k, index_type l)
                {
                    return fun(mesh::MeshEntityIdCoder::pack_index4<IFORM>(i, j, k, l));
                });
    }

    template<typename ...Args> void
    foreach(mesh::MeshZoneTag const &tag, Args &&...args)
    {
        deploy();
        if (tag == mesh::SP_ES_ALL)
        {
            foreach(std::forward<Args>(args)...);
        } else if (tag == mesh::SP_ES_GHOST)
        {
            foreach_ghost(std::forward<Args>(args)...);
        } else
        {
            foreach(m_->mesh_block()->range(entity_type(), tag, DOF), std::forward<Args>(args)...);
        }
    }

    void copy(mesh::EntityIdRange const &r0, this_type const &g)
    {
        UNIMPLEMENTED;
//        r0.foreach([&](mesh::MeshEntityId const &s) { get(s) = g.get(s); });
    }


    virtual void copy(mesh::EntityIdRange const &r0, mesh::DataBlock const &other)
    {
        UNIMPLEMENTED;
//        assert(other.is_a(typeid(this_type)));
//
//        this_type const &g = static_cast<this_type const & >(other);
//
//        copy(r0, static_cast<this_type const & >(other));

    }


};

//namespace traits
//{
//template<typename TV, typename TM, typename ...Others>
//struct reference<Field<TV, TM, Others...>> { typedef Field<TV, TM, Others...> const &type; };
//}
}//namespace simpla







#endif //SIMPLA_FIELD_H
