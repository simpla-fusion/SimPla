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

template<typename TV, typename TManifold, mesh::MeshEntityType IFORM> using field_t= Field<TV, TManifold, index_const<IFORM>>;


template<typename TV, typename TManifold, size_t I>
class Field<TV, TManifold, index_const<I >> :
        public mesh::AttributeView<TV, static_cast<mesh::MeshEntityType >(I)>
{
private:
    static_assert(std::is_base_of<mesh::MeshBlock, TManifold>::value, "TManifold is not derived from MeshBlock");

    typedef Field<TV, TManifold, index_const<I >> this_type;

    typedef mesh::AttributeView<TV, static_cast<mesh::MeshEntityType >(I)> base_type;

    static constexpr mesh::MeshEntityType IFORM = static_cast<mesh::MeshEntityType>(I);


public:
    typedef typename traits::field_value_type<this_type>::type field_value_type;
    typedef TManifold mesh_type;
    typedef TV value_type;
    using base_type::entity_type;

    using base_type::value_type_info;
private:
    typedef typename mesh_type::template data_block_type<TV, IFORM> data_block_type;

    mesh_type const *m_mesh_ = nullptr;
    data_block_type *m_data_ = nullptr;
public:
    Field() {};

    template<typename ...Args>
    Field(mesh_type const *m, Args &&...args) :base_type(m, std::forward<Args>(args)...) {};

    template<typename ...Args>
    Field(std::string const &s, Args &&...args) :base_type(s, std::forward<Args>(args)...)
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

    virtual bool is_a(std::type_info const &t_info) const
    {
        return t_info == typeid(this_type) || base_type::is_a(t_info);
    };


    virtual void clear()
    {
        deploy();
        ASSERT(m_data_ != nullptr);
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

    virtual void sync(mesh::MeshBlock const *other, bool only_ghost = true) { UNIMPLEMENTED; };


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
        apply(_impl::_assign(), other);
        return *this;
    }

    template<typename ...U> inline this_type &
    operator=(Field<U...> const &other)
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

    template<typename TOther> void
    assign(TOther const &v)
    {
        apply(_impl::_assign(), v);
    }

    /* @}*/
private:


    template<typename TOP, typename ...U>
    void apply(TOP const &op, Field<Expression<U...>> const &expr)
    {
        m_data_->for_each(op,
                          [&](index_type i, index_type j, index_type k,
                              index_type n)
                          {
                              return m_mesh_->eval(expr,
                                                   mesh::MeshEntityIdCoder::pack_index(i, j, k, n));
                          });
    }

    template<typename TOP, typename TFun, typename ...U> void
    apply(TOP const &op, TFun const &fun, U &&...args)
    {
        m_data_->for_each(op,
                          [&](index_type i, index_type j, index_type k, index_type n)
                          {
                              return fun(m_mesh_->point(i, j, k, n),
                                         std::forward<U>(args)...);
                          });
    }

    template<typename TOP, typename U> void
    apply(TOP const &op, U const &v, ENABLE_IF(std::is_scalar<U>::value))
    {
        m_data_->for_each(op, [&](index_type i, index_type j, index_type k, index_type n) { return v; });
    }


public:

    void copy(mesh::EntityIdRange const &r0, this_type const &g) {}

    virtual void copy(mesh::EntityIdRange const &r0, mesh::DataBlock const &other) {}


};

namespace traits
{
template<typename TV, typename TM, size_t I>
struct reference<Field<TV, TM, index_const<I> > > { typedef Field<TV, TM, index_const<I> > const &type; };
}
}//namespace simpla







#endif //SIMPLA_FIELD_H
