/**
 * @file attribute.h
 * @author salmon
 * @date 2015-12-16.
 */

#ifndef SIMPLA_ATTRIBUTE_H
#define SIMPLA_ATTRIBUTE_H

#include "../data_model/DataSet.h"
#include "Patch.h"
#include "DataObject.h"


namespace simpla { namespace base
{

class AttributeObject : public base::DataObject
{
public:
    SP_OBJECT_HEAD(AttributeObject, DataObject);

    AttributeObject(std::string const &s_name = "") : m_name_(s_name) { }

    virtual ~AttributeObject() { }

    virtual int center_type() const { return 0; }

    virtual int rank() const { return 0; }

    virtual int extent(int i) const { return 1; }

    virtual data_model::DataSet data_set() = 0;

    virtual data_model::DataSet data_set() const = 0;


    void swap(AttributeObject &other)
    {
        std::swap(m_name_, other.m_name_);
        base_type::swap(other);
    }

public:
    virtual std::string const &name() const
    {
        if (m_parent_.expired()) { return m_name_; }
        else { return m_parent_.lock()->name(); }
    }

    virtual std::shared_ptr<AttributeObject> parent() { return m_parent_.lock(); }

private:
    std::weak_ptr<AttributeObject> m_parent_;
    std::string m_name_;
};

template<typename TMesh>
class AttributeEntity : public AttributeObject
{
    typedef TMesh mesh_type;

    typedef typename mesh_type::id_type id_type;

    typedef AttributeEntity<mesh_type> this_type;
    typedef AttributeObject base_type;

public:
    AttributeEntity(mesh_type &m, std::string const &s_name = "")
            : AttributeObject(s_name), m_mesh_(&m), m_const_mesh_(&m) { }

    AttributeEntity(mesh_type const &m, std::string const &s_name = "")
            : AttributeObject(s_name), m_mesh_(nullptr), m_const_mesh_(&m) { }

    AttributeEntity(AttributeEntity const &other)
            : m_mesh_(other.m_mesh_), m_const_mesh_(other.m_const_mesh_) { }

    AttributeEntity(AttributeEntity &&other)
            : m_mesh_(other.m_mesh_), m_const_mesh_(other.m_const_mesh_) { }

    virtual ~AttributeEntity() { };

    void swap(this_type &other)
    {
        std::swap(m_mesh_, other.m_mesh_);
        base_type::swap(other);
    }

    AttributeEntity &operator=(AttributeEntity const &other)
    {
        AttributeEntity(other).swap(*this);
        return *this;
    }

    virtual bool is_a(std::type_info const &info) const
    {
        return typeid(this_type) == info || base_type::is_a(info);
    }

    virtual std::string get_class_name() const
    {
        return "AttributeEntity<" + m_mesh_->get_class_name() + ">";
    }


    mesh_type const &mesh() { return *m_const_mesh_; }

    mesh_type const &mesh() const { return *m_const_mesh_; }

    mesh_type &get_mesh()
    {
        if (m_mesh_ == nullptr)
        {
            VERBOSE << ("Can not  modified const Mesh!");
        }
        return *m_mesh_;


    }


private:
    mesh_type *m_mesh_;
    mesh_type const *m_const_mesh_;

};

template<typename TV, int IFORM, typename TMesh>
class Attribute :
        public AttributeEntity<TMesh>,
        public Patch<Attribute<TV, IFORM, TMesh> >
{

private:
    typedef TMesh mesh_type;

    typedef typename mesh_type::id_type id_type;

    typedef Attribute<TV, IFORM, TMesh> this_type;

    typedef AttributeEntity<TMesh> base_type;

    typedef Patch<Attribute<TV, IFORM, TMesh>> patch_policy;
public:
    using base_type::mesh;
    typedef TV value_type;

    template<typename ...Args>
    Attribute(mesh_type &m, Args &&...args) : base_type(m, std::forward<Args>(args)...), m_data_(nullptr) { }

    template<typename ...Args>
    Attribute(mesh_type const &m, Args &&...args) : base_type(m, std::forward<Args>(args)...), m_data_(nullptr) { }

    Attribute(Attribute const &other) : base_type(other), m_data_(other.m_data_) { }

    Attribute(Attribute &&other) : base_type(other), m_data_(other.m_data_) { }

    virtual ~Attribute() { }

    Attribute &operator=(Attribute const &other)
    {
        base_type(other).swap(*this);
        return *this;
    }

    void swap(Attribute &other)
    {
        base_type::swap(other);
        std::swap(m_data_, other.m_data_);
    }

    virtual bool is_a(std::type_info const &info) const
    {
        return typeid(this_type) == info || base_type::is_a(info);
    }

    virtual std::string get_class_name() const
    {
        return "Attribute<" + mesh().get_class_name() + ">";
    }


    virtual this_type &self() { return *this; }

    virtual this_type const &self() const { return *this; }

    virtual int center_type() const { return IFORM; };

    virtual int rank() const { return traits::rank<TV>::value; }

    virtual int extent(int i) const { return traits::seq_value<typename traits::extents<TV>::type>::value[i]; }

    virtual data_model::DataSet data_set() const { return this->mesh().template data_set<TV, IFORM>(m_data_); };

    virtual data_model::DataSet data_set() { return this->mesh().template data_set<TV, IFORM>(m_data_); };

    virtual bool empty() const { return m_data_ == nullptr; }

    virtual void deploy() { if (empty()) { m_data_ = this->mesh().template data<value_type, IFORM>(); }}

    virtual void clear()
    {
        deploy();
        memset(m_data_.get(), 0, this->mesh().template memory_size<IFORM>() * sizeof(value_type));
    }

    virtual void sync()
    {
        auto ds = data_set();
        this->mesh().sync(ds);
    }

    value_type &at(id_type const &s)
    {
        return m_data_.get()[this->mesh().hash(s)];
    }

    value_type const &at(id_type const &s) const
    {
        return m_data_.get()[this->mesh().hash(s)];
    }

    value_type &operator[](id_type const &s) { return at(s); }

    value_type const &operator[](id_type const &s) const { return at(s); }

    typename mesh_type::range_type range() { return this->mesh().template range<IFORM>(); }

    template<typename TRange, typename Func>
    void accept(TRange const &r0, Func const &fun)
    {
        deploy();
        this->mesh().template for_each_value<value_type, IFORM>(*this, r0, fun);
    };

    /// @name AMR @{

    virtual std::shared_ptr<this_type> create_patch(size_t id)
    {
        std::shared_ptr<this_type> res(nullptr);
//@FIXME
//        auto p_mesh = mesh().Patch(id);
//
//        if (p_mesh != nullptr)
//        {
//            res = p_mesh->template create_attribute<TV, IFORM>(AttributeObject::name());
//        }
        return res;
    }

    /// @}
protected:
    std::shared_ptr<value_type> m_data_;

};
}}//namespace simpla { namespace base


#endif //SIMPLA_ATTRIBUTE_H
