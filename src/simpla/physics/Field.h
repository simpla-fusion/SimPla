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
#include <simpla/mesh/MeshCommon.h>

#include "FieldTraits.h"
#include "FieldExpression.h"

namespace simpla
{

template<typename ...> class Field;


template<typename TV, typename TManifold, size_type I, size_type DOF>
class Field<TV, TManifold, index_const<I>, index_const<DOF>> :
        public mesh::AttributeView<TV, static_cast<mesh::MeshEntityType>(I), DOF>
{
private:
    static_assert(std::is_base_of<mesh::MeshBlock, TManifold>::value, "TManifold is not derived from MeshBlock");

    typedef Field<TV, TManifold, index_const<I>, index_const<DOF>> this_type;
    typedef mesh::AttributeView<TV, static_cast<mesh::MeshEntityType>(I), DOF> base_type;
    static constexpr mesh::MeshEntityType IFORM = static_cast<mesh::MeshEntityType>(I);


public:
    typedef TV value_type;
    typedef typename std::conditional<DOF == 1, value_type, nTuple<value_type, DOF> >::type cell_tuple;
    typedef typename std::conditional<(IFORM == mesh::VERTEX || IFORM == mesh::VOLUME),
            cell_tuple, nTuple<cell_tuple, 3> >::type field_value_type;
    typedef TManifold mesh_type;
private:
    typedef typename mesh_type::template data_block_type<TV, static_cast<mesh::MeshEntityType>(IFORM)> data_block_type;

    mesh_type const *m_mesh_ = nullptr;
    data_block_type *m_data_ = nullptr;
public:
    Field() {};

    template<typename ...Args>
    Field(mesh_type const *m, Args &&...args) :base_type(std::forward<Args>(args)...), m_mesh_(m) {};

    template<typename ...Args>
    explicit Field(Args &&...args) :base_type(std::forward<Args>(args)...)
    {
        base_type::attribute()->register_data_block_factroy(
                std::type_index(typeid(mesh_type)),
                [&](const mesh::MeshBlock *m, void *p)
                {
                    return static_cast<mesh_type const *>(m)->template create_data_block<value_type, IFORM>(p);
                });

    };

    virtual ~Field() {}

    Field(this_type const &other) = delete;

    Field(this_type &&other) = delete;

    using base_type::entity_type;
    using base_type::value_type_info;

    virtual bool is_a(std::type_info const &t_info) const
    {
        return t_info == typeid(this_type) || base_type::is_a(t_info);
    };


    virtual void clear()
    {
        deploy();
        m_data_->clear();
    }

    virtual void deploy()
    {
        base_type::deploy();
        m_mesh_ = static_cast<mesh_type const *>(base_type::mesh());
        m_data_ = static_cast<data_block_type *>(base_type::data());
    }

    virtual std::shared_ptr<mesh::DataBlock> clone(const std::shared_ptr<mesh::MeshBlock> &m) const
    {
        // FIXME: new data block is not initialized!!
        return std::dynamic_pointer_cast<mesh::DataBlock>(std::make_shared<data_block_type>());
    };

    /** @name as_function  @{*/
    template<typename ...Args> field_value_type
    gather(Args &&...args) const { return m_mesh_->gather(*this, std::forward<Args>(args)...); }

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
        r0.foreach([&](mesh::MeshEntityId const &s) { get(s) = m_mesh_->eval(expr, s); });

    }

    template<typename TFun, typename ...U> void
    foreach(mesh::EntityIdRange const &r0, std::function<value_type(point_type const &, U const &...)> const &fun,
            U &&...args)
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { get(s) = fun(m_mesh_->point(s), std::forward<U>(args)...); });
    }


    template<typename TFun> void
    foreach(mesh::EntityIdRange const &r0, TFun const &fun,
            ENABLE_IF((std::is_same<typename std::result_of<TFun(point_type const &)>::type, field_value_type>::value))
    )
    {
        deploy();
        r0.foreach(
                [&](mesh::MeshEntityId const &s)
                {
                    get(s) = m_mesh_->template sample<IFORM>(s, fun(m_mesh_->point(s)));
                });
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
        auto b = std::get<0>(m_mesh_->outer_index_box());
        index_type gw[4] = {1, 1, 1, 0};

        m_data_->foreach(
                [&](index_type i, index_type j, index_type k, index_type l)
                {
                    auto s = mesh::MeshEntityIdCoder::pack_index4<IFORM>(i, j, k, l);
                    return m_mesh_->eval(expr, s);
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
                    return fun(m_mesh_->point(i, j, k));
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
                    return m_mesh_->template sample<IFORM>(s, fun(m_mesh_->point(s)));
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
                    return m_mesh_->eval(expr, mesh::MeshEntityIdCoder::pack_index4<IFORM>(i, j, k, l));
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
                    return m_mesh_->template sample<IFORM>(s, fun(m_mesh_->point(s)));
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
            foreach(m_mesh_->range(entity_type(), tag), std::forward<Args>(args)...);
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
