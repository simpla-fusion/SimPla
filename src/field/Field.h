/**
 * @file field.h
 * @author salmon
 * @date 2015-10-16.
 */

#ifndef SIMPLA_FIELD_H
#define SIMPLA_FIELD_H

#include <type_traits>
#include <cassert>

#include "../mesh/MeshAttribute.h"
#include "../data_model/DataSet.h"

#include "FieldTraits.h"


namespace simpla
{

template<typename ...>
class Field;

template<typename TV, typename TManifold, size_t IFORM> using field_t= Field<TV, TManifold, index_const<IFORM>>;


template<typename TV, typename TManifold, size_t IFORM>
class Field<TV, TManifold, index_const<IFORM>> :
        public mesh::MeshAttribute::View,
        public std::enable_shared_from_this<Field<TV, TManifold, index_const<IFORM>>>
{
private:
    static_assert(std::is_base_of<mesh::MeshBase, TManifold>::value, "TManifold is not derived from MeshBase");

    typedef Field<TV, TManifold, index_const<IFORM>> this_type;

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

    static constexpr mesh::MeshEntityType iform = static_cast<mesh::MeshEntityType>(IFORM);


    //create construct

    Field(std::shared_ptr<mesh_type const> m = nullptr)
            : m_holder_(nullptr), m_mesh_(m), m_data_(nullptr), m_range_()
    {
    }

    Field(std::shared_ptr<mesh::MeshBase const> m)
            : m_holder_(nullptr), m_mesh_(std::dynamic_pointer_cast<mesh_type const>(m)), m_data_(nullptr),
              m_range_()
    {
        assert(m->template is_a<mesh_type>());

    }

    Field(std::shared_ptr<base_type> h)
            : m_holder_(h), m_mesh_(nullptr), m_data_(nullptr), m_range_()
    {
    }

    //factory construct
    template<typename TFactory, typename ... Args, typename std::enable_if<TFactory::is_factory>::type * = nullptr>
    Field(TFactory &factory, Args &&...args)
            : m_holder_(std::dynamic_pointer_cast<base_type>(
            factory.template create<this_type>(std::forward<Args>(args)...))),
              m_mesh_(nullptr), m_data_(nullptr), m_range_()
    {
    }


    //copy construct
    Field(this_type const &other)
            : m_holder_(other.m_holder_), m_mesh_(other.m_mesh_), m_data_(other.m_data_), m_range_(other.m_range_)
    {
    }


    // move construct
    Field(this_type &&other)
            : m_holder_(other.m_holder_), m_mesh_(other.m_mesh_), m_data_(other.m_data_), m_range_(other.m_range_)
    {
    }

    virtual ~Field() { }

    virtual bool deploy()
    {
        bool success = false;

        if (m_holder_ == nullptr) {
            if (m_data_ == nullptr) {
                if (m_mesh_ == nullptr) { RUNTIME_ERROR << "mesh is not valid!" << std::endl; }
                else {
                    size_t m_size = m_mesh_->max_hash(entity_type());

                    m_data_ = sp_alloc_array<value_type>(m_size);

                    m_mesh_->range(entity_type()).swap(m_range_);

                }

                success = true;
            }
        }
        else {
            if (m_holder_->is_a<this_type>()) {
                m_holder_->deploy();
                auto self = std::dynamic_pointer_cast<this_type>(m_holder_);
                m_mesh_ = self->m_mesh_;
                m_data_ = self->m_data_;
                m_range_ = self->m_range_;
                success = true;
            }
        }

        return success;
    }

    bool empty() const { return m_holder_ == nullptr && m_data_ == nullptr; }

    virtual void swap(this_type &other)
    {
        base_type::swap(other);
        std::swap(m_mesh_, other.m_mesh_);
        std::swap(m_data_, other.m_data_);
        m_range_.swap(other.m_range_);
    }

    virtual void clear()
    {
        deploy();
        for (auto const &s: m_mesh_->range(entity_type())) { get(s) = 0; }
    }

    virtual std::shared_ptr<base_type> get_view()
    {
        return std::dynamic_pointer_cast<base_type>(this->shared_from_this());
    }

    virtual mesh::MeshEntityType entity_type() const { return static_cast<mesh::MeshEntityType >(IFORM); }

    virtual mesh::MeshEntityRange const &range() const { return m_range_; };

    virtual data_model::DataSet data_set() const
    {
        data_model::DataSet res;

        res.data_type = data_model::DataType::create<value_type>();

        res.data = std::shared_ptr<void>(m_data_.get(), tags::do_nothing());

        std::tie(res.memory_space, res.data_space) = m_mesh_->data_space(entity_type());

        return std::move(res);
    };

    virtual void data_set(data_model::DataSet const &)
    {
        UNIMPLEMENTED;
    };


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
        return mesh_type::interpolate_policy::gather(*m_mesh_, *this, std::forward<Args>(args)...);
    }


    template<typename ...Args>
    field_value_type
    operator()(Args &&...args) const
    {
        return gather(std::forward<Args>(args)...);
    }
    /**@}*/
public:


    this_type select(mesh::MeshEntityRange const &r) const
    {
        this_type res(*this);
        res.m_range_ = r;
        return std::move(res);
    }


    template<typename TSelecter>
    this_type
    select(TSelecter const &pred,
           FUNCTION_REQUIREMENT((std::is_same<typename std::result_of<TSelecter(
                   typename mesh::MeshEntityRange const &)>::type, mesh::MeshEntityRange>::value))
    )
    {
        return std::move(select(pred(m_range_)));
    }

    template<typename TFun>
    this_type &
    apply(mesh::MeshEntityRange const &r, TFun const &op,
          FUNCTION_REQUIREMENT((std::is_same<typename std::result_of<TFun(
                  typename mesh::point_type const &,
                  field_value_type const &)>::type, field_value_type>::value))
    )
    {
        deploy();

        //TODO: need parallelism
        if (!r.empty()) {
            for (auto const &s: r) {
                auto x = m_mesh_->point(s);
                get(s) = m_mesh_->sample(s, op(x, gather(x)));
            }
        }

        return *this;
    }

    template<typename TFun>
    this_type &
    apply(mesh::MeshEntityRange const &r, TFun const &op,
          FUNCTION_REQUIREMENT((std::is_same<typename std::result_of<TFun(
                  typename mesh::point_type const &)>::type, field_value_type>::value))
    )
    {
        deploy();

        //TODO: need parallelism
        if (!r.empty()) { for (auto const &s: r) { get(s) = m_mesh_->sample(s, op(m_mesh_->point(s))); }}

        return *this;
    }

    template<typename TFun>
    this_type &
    apply(mesh::MeshEntityRange const &r, TFun const &op,
          FUNCTION_REQUIREMENT(
                  (std::is_same<typename std::result_of<TFun(mesh::MeshEntityId const &)>::type, value_type>::value)
          )
    )
    {
        deploy();

        //TODO: need parallelism
        if (!r.empty()) { for (auto const &s: r) { get(s) = op(s); }}

        return *this;
    }

    template<typename TFun>
    this_type &
    apply(mesh::MeshEntityRange const &r, TFun const &op,
          FUNCTION_REQUIREMENT(
                  (std::is_same<typename std::result_of<TFun(value_type &)>::type, void>::value)
          )
    )
    {
        deploy();

        //TODO: need parallelism
        if (!r.empty()) { for (auto const &s: r) { op(get(s)); }}

        return *this;
    }


    template<typename TFun>
    this_type &
    apply(mesh::MeshEntityRange const &r, TFun const &f,
          FUNCTION_REQUIREMENT((traits::is_indexable<TFun, typename mesh::MeshEntityId>::value))
    )
    {
        deploy();

        //TODO: need parallelism
        if (!r.empty()) { for (auto const &s: r) { get(s) = f[s]; }}

        return *this;
    }

private:

    template<typename TOP, typename Other>
    this_type &
    apply_expr(TOP const &op, Other const &other)
    {
        deploy();
        //TODO: need parallelism
        if (!m_range_.empty()) { for (auto const &s: m_range_) { op(get(s), m_mesh_->eval(other, s)); }}
        return *this;
    }


    std::shared_ptr<mesh_type const> m_mesh_; // @FIXME [severity: low, potential] here is a potential risk, mesh maybe destroyed before data;
    std::shared_ptr<value_type> m_data_;
    mesh::MeshEntityRange m_range_;

    std::shared_ptr<base_type> m_holder_;

};


}//namespace simpla

#endif //SIMPLA_FIELD_H
