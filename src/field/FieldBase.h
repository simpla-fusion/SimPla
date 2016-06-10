/**
 * @file field_dense.h
 *
 *  Created on: @date{ 2015-1-30}
 *      @author: salmon
 */

#ifndef FIELD_DENSE_H_
#define FIELD_DENSE_H_


#include "../gtl/type_traits.h"
#include "../mesh/MeshBase.h"
#include "../mesh/MeshAttribute.h"
#include "../data_model/DataSet.h"

#include "FieldTraits.h"

namespace simpla
{


template<typename TV, typename TManifold, size_t IFORM>
class Field<TV, TManifold, index_const<IFORM>> :
        public mesh::MeshAttribute::View,
        public std::enable_shared_from_this<Field<TV, TManifold, index_const<IFORM>>>
{
private:
    static_assert(std::is_base_of<mesh::MeshBase, TManifold>::value, "TManifold is not derived from MeshBase");

    typedef Field<TV, TManifold, index_const<IFORM>> this_type;

    static constexpr mesh::MeshEntityType iform = static_cast<mesh::MeshEntityType>(IFORM);

    typedef typename mesh::MeshAttribute::View base_type;
public:

    virtual bool is_a(std::type_info const &info) const { return typeid(this_type) == info; }

    virtual std::string get_class_name() const { return class_name(); }

    static std::string class_name()
    {
        return std::string("Field<") +
               traits::type_id<value_type, mesh_type, index_const<IFORM>>::name() + ">";
    }


public:

    typedef TManifold mesh_type;

    typedef TV value_type;

    typedef typename traits::field_value_type<this_type>::type field_value_type;



    //create construct

    Field(mesh_type const *m = nullptr) : m_holder_(nullptr), m_mesh_(m), m_data_(nullptr) { }

    Field(mesh::MeshBase const *m)
            : m_holder_(nullptr), m_mesh_(dynamic_cast<mesh_type const *>(m)), m_data_(nullptr)
    {
        assert(m->template is_a<mesh_type>());
    }

    Field(std::shared_ptr<base_type> h) : m_holder_(h), m_mesh_(nullptr), m_data_(nullptr) { }

    //factory construct
    template<typename TFactory, typename ... Args, typename std::enable_if<TFactory::is_factory>::type * = nullptr>
    Field(TFactory &factory, Args &&...args)
            : m_holder_(std::dynamic_pointer_cast<base_type>(
            factory.template create<this_type>(std::forward<Args>(args)...))),
              m_mesh_(nullptr), m_data_(nullptr)
    {
    }


    //copy construct
    Field(this_type const &other)
            : m_holder_(other.m_holder_), m_mesh_(other.m_mesh_), m_data_(other.m_data_)
    {
    }


    // move construct
    Field(this_type &&other)
            : m_holder_(other.m_holder_), m_mesh_(other.m_mesh_), m_data_(other.m_data_)
    {
    }

    virtual ~Field() { }

    std::ostream &print(std::ostream &os, int indent) const
    {
//        os << std::setw(indent + 1) << " Type= " << get_class_name() << ", ";
        return os;
    };

    virtual mesh::MeshBase const *get_mesh() const { return dynamic_cast<mesh::MeshBase const *>(m_mesh_); };

    virtual bool set_mesh(mesh::MeshBase const *m)
    {
        UNIMPLEMENTED;
        assert(m->is_a<mesh_type>());
        m_mesh_ = dynamic_cast<mesh_type const * >(m);
        return false;
    }

    virtual mesh::MeshEntityRange entity_id_range(mesh::MeshEntityStatus entityStatus = mesh::VALID) const
    {
        assert(m_mesh_ != nullptr);
        return m_mesh_->range(entity_type(), entityStatus);
    }

    virtual bool deploy()
    {
        bool success = false;

        if (m_holder_ == nullptr)
        {
            if (m_data_ == nullptr)
            {
                if (m_mesh_ == nullptr) { RUNTIME_ERROR << "get_mesh is not valid!" << std::endl; }
                else
                {
                    size_t m_size = m_mesh_->max_hash(entity_type());

                    m_data_ = sp_alloc_array<value_type>(m_size);

                }

                success = true;
            }
        }
        else
        {
            if (m_holder_->is_a<this_type>())
            {
                m_holder_->deploy();
                auto self = std::dynamic_pointer_cast<this_type>(m_holder_);
                m_mesh_ = self->m_mesh_;
                m_data_ = self->m_data_;
                success = true;
            }
        }

        return success;
    }

    bool empty() const { return m_holder_ == nullptr && m_data_ == nullptr; }

    virtual bool is_valid() const { return !empty(); }

    virtual void swap(base_type &other)
    {
        assert(other.is_a<this_type>());
        swap(dynamic_cast<this_type &>(other));
    };

    virtual void swap(this_type &other)
    {
        std::swap(m_mesh_, other.m_mesh_);
        std::swap(m_data_, other.m_data_);
    }

    virtual void clear()
    {
        deploy();
        parallel::parallel_foreach(m_mesh_->range(entity_type()), [&](mesh::MeshEntityId const &s) { get(s) = 0; });
    }


    virtual mesh::MeshEntityType entity_type() const { return static_cast<mesh::MeshEntityType >(IFORM); }

    virtual data_model::DataSet dataset() const
    {
        data_model::DataSet res;

        res.data_type = data_model::DataType::create<value_type>();

        res.data = std::shared_ptr<void>(m_data_.get(), tags::do_nothing());

        std::tie(res.memory_space, res.data_space) = m_mesh_->data_space(entity_type());

        return res;
    };

    virtual void dataset(data_model::DataSet const &)
    {
        UNIMPLEMENTED;
    };

    virtual void dataset(mesh::MeshEntityRange const &, data_model::DataSet const &)
    {
        UNIMPLEMENTED;
    }

    virtual data_model::DataSet dataset(mesh::MeshEntityRange const &) const
    {
        UNIMPLEMENTED;
        return data_model::DataSet();
    }


public:
/** @name as_array
 *  @{*/
    this_type &operator=(this_type const &other) { return apply_expr(_impl::_assign(), other); }

    template<typename Other>
    inline this_type &
    operator=(Other const &other) { return apply_expr(_impl::_assign(), other); }

    template<typename Other>
    inline this_type &
    operator+=(Other const &other) { return apply_expr(_impl::plus_assign(), other); }

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
        assert(!empty());

        return m_data_.get()[m_mesh_->hash(s)];
    }

    inline value_type const &get(mesh::MeshEntityId const &s) const
    {
        size_t n = m_mesh_->hash(s);
        return m_data_.get()[n];
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

    template<typename TFun> this_type &
    apply(mesh::MeshEntityRange const &r0, TFun const &op,
          FUNCTION_REQUIREMENT((std::is_same<typename std::result_of<TFun(
                  typename mesh::point_type const &,
                  field_value_type const &)>::type, field_value_type>::value))
    )
    {
        deploy();

        if (!r0.empty())
        {
            parallel::parallel_foreach(
                    r0, [&](typename mesh::MeshEntityId const &s)
                    {
                        auto x = m_mesh_->point(s);
                        get(s) = m_mesh_->template sample<IFORM>(s, op(x, gather(x)));
                    }
            );
        }

        return *this;
    }

    template<typename TFun> this_type &
    apply(mesh::MeshEntityRange const &r0, TFun const &op,
          FUNCTION_REQUIREMENT((std::is_same<typename std::result_of<TFun(
                  typename mesh::point_type const &)>::type, field_value_type>::value))
    )
    {
        deploy();

        if (!r0.empty())
        {
            parallel::parallel_foreach(
                    r0, [&](typename mesh::MeshEntityId const &s)
                    {
                        auto v = op(m_mesh_->point(s));
                        get(s) = m_mesh_->template sample<IFORM>(s, v);

                    }
            );
        }

        return *this;
    }

    template<typename TFun> this_type &
    apply(mesh::MeshEntityRange const &r0, TFun const &op,
          FUNCTION_REQUIREMENT(
                  (std::is_same<typename std::result_of<TFun(mesh::MeshEntityId const &)>::type, value_type>::value)
          ))
    {
        deploy();

        if (!r0.empty())
        {
            parallel::parallel_foreach(r0, [&](typename mesh::MeshEntityId const &s) { get(s) = op(s); });
        }
        return *this;
    }

    template<typename TFun> this_type &
    apply(mesh::MeshEntityRange const &r0, TFun const &op,
          FUNCTION_REQUIREMENT(
                  (std::is_same<typename std::result_of<TFun(value_type &)>::type, void>::value)
          ))
    {
        deploy();

        if (!r0.empty())
        {
            parallel::parallel_foreach(r0, [&](typename mesh::MeshEntityId const &s) { op(get(s)); });
        }

        return *this;
    }


    template<typename TFun> this_type &
    apply(mesh::MeshEntityRange const &r0, TFun const &f,
          FUNCTION_REQUIREMENT((traits::is_indexable<TFun, typename mesh::MeshEntityId>::value)))
    {

        deploy();

        if (!r0.empty())
        {
            parallel::parallel_foreach(r0, [&](typename mesh::MeshEntityId const &s) { get(s) = f[s]; });
        }

        return *this;
    }

    template<typename TOP> this_type &
    apply(TOP const &op)
    {
        deploy();

        apply(m_mesh_->range(entity_type(), mesh::NON_LOCAL), op);
        base_type::nonblocking_sync();
        apply(m_mesh_->range(entity_type(), mesh::LOCAL), op);
        base_type::wait();
        return *this;
    }

private:

    template<typename TOP, typename Other> this_type &
    apply_expr(mesh::MeshEntityRange const &r, TOP const &op, Other const &other)
    {
        if (!r.empty())
        {
            parallel::parallel_foreach(
                    r, [&](typename mesh::MeshEntityId const &s)
                    {
                        op(get(s), m_mesh_->eval(other, s));
                    }

            );
        }
        return *this;
    }


    template<typename TOP, typename Other> this_type &
    apply_expr(TOP const &op, Other const &other)
    {
        deploy();

        apply_expr(m_mesh_->range(entity_type(), mesh::NON_LOCAL), op, other);
        base_type::nonblocking_sync();
        apply_expr(m_mesh_->range(entity_type(), mesh::LOCAL), op, other);
        base_type::wait();
        return *this;
    }

protected:
    mesh_type const *m_mesh_;
    std::shared_ptr<value_type> m_data_;
    std::shared_ptr<base_type> m_holder_;

};

}// namespace simpla

#endif /* FIELD_DENSE_H_ */
