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
#include <simpla/mesh/Attribute.h>

#include "Bundle.h"
#include "FieldTraits.h"
#include "FieldExpression.h"
#include "schemes/CalculusPolicy.h"
#include "schemes/InterpolatePolicy.h"


namespace simpla
{


template<typename ...> class Field;


template<typename TV, typename TM, size_type I, size_type DOF>
class Field<TV, TM, index_const<I>, index_const<DOF>> :
        public mesh::Bundle<TV, static_cast<mesh::MeshEntityType>(I), DOF>
{
private:
    static constexpr mesh::MeshEntityType IFORM = static_cast<mesh::MeshEntityType>(I);
    typedef Field<TV, TM, index_const<I>, index_const<DOF>> this_type;
    typedef mesh::Bundle<TV, static_cast<mesh::MeshEntityType>(I), DOF> base_type;
public:
    typedef TV value_type;
    typedef TM mesh_type;
    typedef typename std::conditional<DOF == 1, value_type, nTuple<value_type, DOF> >::type cell_tuple;
    typedef typename std::conditional<(IFORM == mesh::VERTEX || IFORM == mesh::VOLUME),
            cell_tuple, nTuple<cell_tuple, 3> >::type field_value_type;

private:
    typedef typename mesh_type::template data_block_type<TV, IFORM, DOF> data_block_type;

    data_block_type *m_data_;
    mesh_type const *m_mesh_;

public:

    typedef manifold::schemes::CalculusPolicy<mesh_type> calculus_policy;

    typedef manifold::schemes::InterpolatePolicy<mesh_type> interpolate_policy;

    template<typename ...Args>
    explicit Field(Args &&...args):
            base_type(std::forward<Args>(args)...),
            m_mesh_(nullptr),
            m_data_(nullptr) {};


    virtual ~Field() {}

    Field(this_type const &other) = delete;

    Field(this_type &&other) = delete;


    virtual bool is_a(std::type_info const &t_info) const
    {
        return t_info == typeid(this_type) || base_type::is_a(t_info);
    };

    bool is_valid() const
    {
        return m_data_ != nullptr
               && m_mesh_ != nullptr
               && m_data_->is_valid();
    };

    bool empty() const { return m_data_->empty() && m_mesh_ != nullptr; };

    using base_type::entity_type;

    using base_type::value_type_info;

    using base_type::dof;

    virtual std::shared_ptr<mesh::DataBlock>
    create_data_block(std::shared_ptr<mesh::MeshBlock> const &m, value_type *p = nullptr) const
    {
        return data_block_type::create(m, p);
    };

    virtual void update()
    {
        base_type::update();
        m_mesh_ = base_type::template mesh_as<mesh_type>();
        m_data_ = base_type::template data_as<data_block_type>();
    }


    /** @name as_function  @{*/
    template<typename ...Args> field_value_type
    gather(Args &&...args) const { return m_mesh_->mesh_block()->gather(*this, std::forward<Args>(args)...); }

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

    virtual value_type &get(mesh::MeshEntityId const &s) { return m_data_->get(s); }

    virtual value_type const &get(mesh::MeshEntityId const &s) const { return m_data_->get(s); }

    virtual value_type &
    get(index_type i, index_type j, index_type k = 0, index_type l = 0) { return m_data_->get(i, j, k, l); }

    virtual value_type const &
    get(index_type i, index_type j, index_type k = 0, index_type l = 0) const { return m_data_->get(i, j, k, l); }

    template<typename ...U> inline this_type &
    operator=(Field<U...> const &other)
    {
        assign(other);
        return *this;
    }

    template<typename Other> inline this_type &
    operator=(Other const &other)
    {
        assign(other);
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

    inline this_type &
    operator=(this_type const &other)
    {

        assign(other);
        return *this;

    }
    /* @}*/
private:

public:

    template<typename TFun> void
    assign(TFun const &fun, mesh::EntityIdRange const &r0,
           typename std::result_of<TFun(point_type const &)>::type *p = nullptr)
    {
        update();
        r0.foreach([&](mesh::MeshEntityId const &s)
                   {
                       interpolate_policy::assign(*this, *m_mesh_, s, fun(m_mesh_->point(s)));
                   });

    }

    template<typename U> void
    assign(U const &v, mesh::EntityIdRange const &r0,
           ENABLE_IF((std::is_convertible<U, value_type>::value || std::is_same<U, field_value_type>::value)))
    {
        update();

        r0.foreach([&](mesh::MeshEntityId const &s)
                   {
                       interpolate_policy::assign(*this, *m_mesh_, s, v);
                   });

    }

    typedef mesh::MeshEntityIdCoder M;

    void assign(this_type const &other, mesh::EntityIdRange const &r0)
    {
        update();

        r0.foreach([&](mesh::MeshEntityId const &s)
                   {
                       for (int i = 0; i < DOF; ++i) { get(M::sw(s, i)) = other.get(M::sw(s, i)); }
                   });

    }

    template<typename ...U>
    void assign(Field<Expression<U...>> const &expr, mesh::EntityIdRange const &r0)
    {
        update();

        r0.foreach([&](mesh::MeshEntityId const &s)
                   {
                       for (int i = 0; i < DOF; ++i)
                       {
                           get(M::sw(s, i)) = calculus_policy::eval(*m_mesh_, expr, M::sw(s, i));
                       }
                   });

    }


    template<typename Other> void
    assign(Other const &other, mesh::MeshZoneTag const &tag = mesh::SP_ES_ALL)
    {
        update();
        if (tag == mesh::SP_ES_ALL)
        {
            assign(other, m_data_->range());
        } else
        {
            assign(other, m_mesh_->mesh_block()->range(entity_type(), tag));
        }
    }

    void copy(mesh::EntityIdRange const &r0, this_type const &g)
    {
        UNIMPLEMENTED;
//        r0.assign([&](mesh::MeshEntityId const &s) { get(s) = g.get(s); });
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


}//namespace simpla







#endif //SIMPLA_FIELD_H
