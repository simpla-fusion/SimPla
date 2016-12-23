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

#include <simpla/toolbox/Log.h>
#include <simpla/mesh/Attribute.h>

#include "simpla/manifold/schemes/CalculusPolicy.h"
#include "simpla/manifold/schemes/InterpolatePolicy.h"

#include "Algebra.h"
#include "nTuple.h"

namespace simpla
{
{
template<typename TV, typename TM, size_type IFORM, size_type DOF>
class Field<TV, TM, IFORM, DOF> : public mesh::Attribute
{
private:
    typedef Field<TV, TM, IFORM, DOF> field_type;

SP_OBJECT_HEAD(field_type, mesh::Attribute);
public:

    typedef TV value_type;
    typedef TM mesh_type;

    typedef std::conditional_t<DOF == 1, value_type, nTuple<value_type, DOF> > cell_tuple;
    typedef std::conditional_t<(IFORM == VERTEX || IFORM == VOLUME),
            cell_tuple, nTuple<cell_tuple, 3> > field_value;

private:
    typedef typename mesh_type::template data_block_type<TV, IFORM, DOF> data_block_type;

    data_block_type *m_data_;
    mesh_type const *m_mesh_;

public:

    typedef manifold::schemes::CalculusPolicy<mesh_type> calculus_policy;

    typedef manifold::schemes::InterpolatePolicy<mesh_type> interpolate_policy;

    template<typename ...Args>
    explicit Field(Args &&...args):
            base_type(std::forward<Args>(args)  ...),
            m_mesh_(nullptr),
            m_data_(nullptr) {};


    virtual ~Field() {}

    Field(this_type const &other) = delete;

    Field(this_type &&other) = delete;


    virtual size_type entity_type() const { return IFORM; };

    virtual std::type_info const &value_type_info() const { return typeid(typename traits::value_type<TV>::type); };

    virtual size_type dof() const { return DOF; };

    virtual std::shared_ptr<mesh::DataBlock>
    create_data_block(std::shared_ptr<mesh::MeshBlock> const &m, value_type *p = nullptr) const
    {
        return data_block_type::create(m, p);
    };

    virtual void pre_process()
    {
        if (base_type::is_valid()) { return; } else { base_type::pre_process(); }

//        m_mesh_ = self_type::mesh_as<mesh_type>();
//        m_data_ = self_type::data_as<data_block_type>();
        ASSERT(m_data_ != nullptr);
        ASSERT(m_mesh_ != nullptr);

    }

    virtual void post_process()
    {
        if (!base_type::is_valid()) { return; } else { base_type::post_process(); }

        m_mesh_ = nullptr;
        m_data_ = nullptr;

    }

    /** @name as_function  @{*/
    template<typename ...Args> field_value
    gather(Args &&...args) const { return m_mesh_->gather(*this, std::forward<Args>(args)...); }

    template<typename ...Args> field_value
    operator()(Args &&...args) const { return gather(std::forward<Args>(args)...); }

    /**@}*/

    /** @name as_array   @{*/

    virtual value_type &
    get(mesh::MeshEntityId s) { return m_data_->get(mesh::MeshEntityIdCoder::unpack_index4(s, DOF)); }

    virtual value_type const &
    get(mesh::MeshEntityId s) const { return m_data_->get(mesh::MeshEntityIdCoder::unpack_index4(s, DOF)); }


    virtual value_type &
    get(index_type i, index_type j, index_type k = 0, index_type l = 0) { return m_data_->get(i, j, k, l); }

    virtual value_type const &
    get(index_type i, index_type j, index_type k = 0, index_type l = 0) const { return m_data_->get(i, j, k, l); }
//
//    template<typename ...Args>
//    inline value_type &operator()(Args &&...args) { return m_data_block_holder_->get(std::forward<Args>(args)...); }
//
//    template<typename ...Args>
//    inline value_type const &operator()(Args &&...args) const { return m_data_block_holder_->get(std::forward<Args>(args)...); }

    template<typename TI>
    inline value_type &operator[](TI const &s) { return get(s); }

    template<typename TI>
    inline value_type const &operator[](TI const &s) const { return get(s); }


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
           std::result_of_t<TFun(point_type const &)> *p = nullptr)
    {
        pre_process();
        r0.foreach([&](mesh::MeshEntityId const &s)
                   {
                       interpolate_policy::assign(*this, *m_mesh_, s, fun(m_mesh_->point(s)));
                   });

    }

    template<typename U> void
    assign(U const &v, mesh::EntityIdRange const &r0,
           ENABLE_IF((std::is_convertible<U, value_type>::value || std::is_same<U, field_value>::value)))
    {
        pre_process();

        r0.foreach([&](mesh::MeshEntityId const &s)
                   {
                       interpolate_policy::assign(*this, *m_mesh_, s, v);
                   });

    }

    typedef mesh::MeshEntityIdCoder M;

    void assign(this_type const &other, mesh::EntityIdRange const &r0)
    {
        pre_process();

        r0.foreach([&](mesh::MeshEntityId const &s)
                   {
                       for (int i = 0; i < DOF; ++i) { get(M::sw(s, i)) = other.get(M::sw(s, i)); }
                   });

    }

    template<typename ...U>
    void assign(Expression<U...> const &expr, mesh::EntityIdRange const &r0)
    {
        pre_process();

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
        pre_process();
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

}//namespace simpla//namespace algebra{







#endif //SIMPLA_FIELD_H
