/**
 * @file field_dense.h
 *
 *  Created on: @date{ 2015-1-30}
 *      @author: salmon
 */

#ifndef FIELD_DENSE_H_
#define FIELD_DENSE_H_

#include "SIMPLA_config.h"
#include "../toolbox/type_traits.h"
#include "../toolbox/DataSet.h"
#include "../mesh/MeshCommon.h"
#include "../mesh/Block.h"
#include "../mesh/Attribute.h"

namespace simpla
{
template<typename ...>
struct Field;


template<typename TV, typename TManifold, size_t IFORM>
class Field<TV, TManifold, std::integral_constant<size_t, IFORM>> : public mesh::Attribute
{
private:
    static_assert(std::is_base_of<mesh::Block, TManifold>::value, "TManifold is not derived from Block");

    typedef Field<TV, TManifold, std::integral_constant<size_t, IFORM>> this_type;

    static constexpr mesh::MeshEntityType iform = static_cast<mesh::MeshEntityType>(IFORM);

    typedef typename mesh::Attribute base_type;
public:
    typedef TManifold mesh_type;
    typedef TV value_type;
protected:
    mesh_type const *m_mesh_;
    std::shared_ptr<void> m_data_holder_;
//    value_type *m_data_root_ptr_;
public:


    typedef typename traits::field_value_type<this_type>::type field_value_type;

    Field() : base_type(), m_mesh_(nullptr), m_data_holder_(nullptr)
//            , m_data_root_ptr_(nullptr)
    {}

    //create construct
    Field(std::shared_ptr<mesh::Block const> m) : Field(static_cast<mesh_type const *>(m.get())) {}

    Field(mesh::Block const *m) : Field(static_cast<mesh_type const *>(m)) {}

    Field(mesh_type const *m) : m_mesh_(m), m_data_holder_(nullptr)
//            , m_data_root_ptr_(nullptr)
    {}

    //copy construct
    Field(this_type const &other) : m_mesh_(other.m_mesh_), m_data_holder_(other.m_data_holder_)
//            ,m_data_root_ptr_(other.m_data_root_ptr_)
    {}

//    //factory construct
//    template<typename TFactory, typename std::enable_if<TFactory::is_factory>::type * = nullptr>
//    Field(TFactory &factory, std::string const &s_name = "") : Field(factory.mesh())
//    {
//        if (s_name != "") { factory.add_attribute(this, s_name); }
//    }


    virtual ~Field() {}

    virtual void swap(this_type &other)
    {
        std::swap(m_mesh_, other.m_mesh_);
//        std::swap(m_data_root_ptr_, other.m_data_root_ptr_);
        std::swap(m_data_holder_, other.m_data_holder_);
    }

    virtual void deploy()
    {
        if (m_data_holder_ == nullptr) { m_data_holder_ = sp_alloc_memory(size_in_byte()); }

//        if (m_data_root_ptr_ == nullptr)
//        {
//            if (m_data_holder_ == nullptr)
//            {
//                m_data_holder_ = sp_alloc_memory(size_in_byte());
//            }
//
//            m_data_root_ptr_ = reinterpret_cast<value_type *>(m_data_holder_.get());
//
//        }
    }

    virtual void clear()
    {
        deploy();
        memset(m_data_holder_.get(), 0, size_in_byte());
    }

    virtual std::ostream &print(std::ostream &os, int indent) const
    {
//        os << std::setw(indent + 1) << " Type= " << get_class_name() << ", ";
        return os;
    };

    virtual bool is_a(std::type_info const &info) const { return typeid(this_type) == info; }

    virtual std::string get_class_name() const { return class_name(); }

    static std::string class_name()
    {
        return std::string("Field<") +
               traits::type_id<value_type>::name() + "," +
               traits::type_id<mesh_type>::name() + "," +
               traits::type_id<index_const<IFORM>>::name()
               + ">";
    }

    virtual bool is_valid() const { return m_mesh_ != nullptr; }

    virtual bool empty() const { return (!is_valid()) || (m_data_holder_ == nullptr); }


    virtual mesh::EntityRange
    entity_id_range(mesh::MeshEntityStatus entityStatus = mesh::SP_ES_OWNED) const
    {
        assert(is_valid());
        return select(m_mesh_, entity_type(), entityStatus);
    }

    virtual size_type entity_size_in_byte() const { return sizeof(value_type); }

    virtual mesh::MeshEntityType entity_type() const { return static_cast<mesh::MeshEntityType >(IFORM); }

    virtual size_type size_in_byte() const
    {
        assert(is_valid());
        return m_mesh_->size() * entity_size_in_byte() * ((iform == mesh::VERTEX || iform == mesh::VOLUME) ? 1 : 3);
    }

    virtual mesh::Block const *mesh() const { return m_mesh_; }

    virtual std::shared_ptr<void> data() { return m_data_holder_; }

    virtual std::shared_ptr<const void> data() const { return m_data_holder_; }

    virtual toolbox::DataType data_type() const { return toolbox::DataType::create<value_type>(); }

    virtual toolbox::DataSet dataset(mesh::MeshEntityStatus status = mesh::SP_ES_OWNED) const
    {
        toolbox::DataSet res;

        res.data_type = toolbox::DataType::create<value_type>();

        res.data = m_data_holder_;

        std::tie(res.memory_space, res.data_space) = m_mesh_->data_space(entity_type(), status);

        return res;
    };

    virtual void dataset(toolbox::DataSet const &) { UNIMPLEMENTED; };


    virtual void sync(bool is_blocking = true) {};


public:
/** @name as_array
 *  @{*/
    this_type &operator=(this_type const &other) { return apply_expr(_impl::_assign(), other); }

    template<typename Other>
    inline this_type &
    operator=(Other const &other) { return apply_expr(_impl::_assign(), other); }

    template<typename Other>
    inline this_type &
    operator+=(Other const &other)
    {

        m_mesh_->for_each(entity_type(), [&](mesh::MeshEntityId const &s) { get(s) += m_mesh_->eval(other, s); });
        return *this;
    }

    template<typename Other>
    inline this_type &
    operator-=(Other const &other) { return apply_expr(_impl::minus_assign(), other); }

    template<typename Other>
    inline this_type &
    operator*=(Other const &other) { return apply_expr(_impl::multiplies_assign(), other); }

    template<typename Other>
    inline this_type &
    operator/=(Other const &other) { return apply_expr(_impl::divides_assign(), other); }

    inline value_type &get(mesh::MeshEntityId const &s)
    {
        return reinterpret_cast<value_type *>(m_data_holder_.get())[m_mesh_->hash(s)];
    }

    inline value_type const &get(mesh::MeshEntityId const &s) const
    {
        return reinterpret_cast<value_type *>(m_data_holder_.get())[m_mesh_->hash(s)];
    }

    inline value_type &operator[](mesh::MeshEntityId const &s) { return get(s); }

    inline value_type const &operator[](mesh::MeshEntityId const &s) const { return get(s); }
/* @}*/
public:
/** @name as_function
 *  @{*/

    template<typename ...Args>
    field_value_type
    gather(Args &&...args) const
    {
        return m_mesh_->gather(*this, std::forward<Args>(args)...);
    }


    template<typename ...Args>
    field_value_type
    operator()(Args &&...args) const
    {
        return gather(std::forward<Args>(args)...);
    }
    /**@}*/
public:

    template<typename TFun>
    this_type &
    apply(mesh::EntityRange const &r0, TFun const &op,
    CHECK_FUNCTION_SIGNATURE(field_value_type, TFun(point_type const &, field_value_type const &))
    )
    {
        deploy();


        r0.foreach(
                [&](mesh::MeshEntityId const &s)
                {
                    auto x = m_mesh_->point(s);
                    get(s) = m_mesh_->template sample<IFORM>(s, op(x, gather(x)));
                }
        );


        return *this;
    }

    template<typename TFun>
    this_type &
    apply(mesh::EntityRange const &r0, TFun const &op,
    CHECK_FUNCTION_SIGNATURE(field_value_type, TFun(point_type const &))
    )
    {
        deploy();


        r0.foreach(
                [&](mesh::MeshEntityId const &s)
                {
                    auto v = op(m_mesh_->point(s));
                    get(s) = m_mesh_->template sample<IFORM>(s, v);

                }
        );


        return *this;
    }

    template<typename TFun>
    this_type &
    apply(mesh::EntityRange const &r0, TFun const &op,
          CHECK_FUNCTION_SIGNATURE(value_type, TFun(mesh::MeshEntityId const &))
    )
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { get(s) = op(s); });
        return *this;
    }

    template<typename TFun>
    this_type &
    apply(mesh::EntityRange const &r0, TFun const &op,
          CHECK_FUNCTION_SIGNATURE(void, TFun(value_type & )))
    {
        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { op(get(s)); });
        return *this;
    }


    template<typename TFun>
    this_type &
    apply(mesh::EntityRange const &r0, TFun const &f,
          ENABLE_IF((traits::is_indexable<TFun, mesh::MeshEntityId>::value)))
    {

        deploy();
        r0.foreach([&](mesh::MeshEntityId const &s) { get(s) = f[s]; });
        return *this;
    }

    template<typename TOP>
    this_type &
    apply(TOP const &op)
    {
        deploy();

//        apply(m_mesh_->range(entity_type(), mesh::SP_ES_NON_LOCAL), op);
//        base_type::nonblocking_sync();
//        apply(m_mesh_->range(entity_type(), mesh::SP_ES_LOCAL), op);
//        base_type::wait();
        apply(m_mesh_->range(entity_type(), mesh::SP_ES_VALID), op);

        return *this;
    }

    template<typename Other>
    this_type &
    fill(Other const &other)
    {
        this->deploy();

        entity_id_range(mesh::SP_ES_ALL).foreach([&](mesh::MeshEntityId const &s) { get(s) = other; });

        return *this;
    }

private:

    template<typename TOP, typename Other>
    this_type &
    apply_expr(mesh::EntityRange const &r, TOP const &op, Other const &other)
    {
        r.foreach([&](mesh::MeshEntityId const &s) { op(get(s), m_mesh_->eval(other, s)); });
        return *this;
    }


    template<typename TOP, typename Other>
    this_type &
    apply_expr(TOP const &op, Other const &other)
    {
        deploy();

//        apply_expr(m_mesh_->range(entity_type(), mesh::SP_ES_NON_LOCAL), op, other);
//        base_type::nonblocking_sync();
//        apply_expr(m_mesh_->range(entity_type(), mesh::SP_ES_LOCAL), op, other);
//        base_type::wait();

        apply_expr(entity_id_range(mesh::SP_ES_VALID), op, other);
        return *this;
    }


};

}// namespace simpla

#endif /* FIELD_DENSE_H_ */
