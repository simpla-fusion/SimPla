//
// Created by salmon on 16-10-20.
//

#ifndef SIMPLA_ATTRIBUTE_H
#define SIMPLA_ATTRIBUTE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/concept/Object.h>
#include <simpla/concept/Serializable.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Configurable.h>
#include <simpla/design_pattern/Observer.h>

#include "DataBlock.h"

namespace simpla { namespace mesh
{
class Patch;

class Attribute;

struct AttributeDesc : public concept::Configurable, public Object
{
    template<typename ...Args>
    AttributeDesc(Args &&...args) { concept::Configurable::db.parse(std::forward<Args>(args)...); }

    virtual ~AttributeDesc() {}

    virtual std::type_index value_type_index() const =0;

    virtual std::type_info const &value_type_info() const =0;

    virtual size_type entity_type() const =0;

    virtual size_type dof() const =0;
};

template<typename TV, MeshEntityType IFORM, size_type DOF>
struct AttributeDescTemp : public AttributeDesc
{
    template<typename ...Args>
    AttributeDescTemp(Args &&...args) :AttributeDesc(std::forward<Args>(args)...) {}

    virtual ~AttributeDescTemp() {}

    virtual std::type_index value_type_index() const { return std::type_index(value_type_info()); };

    virtual std::type_info const &value_type_info() const { return (typeid(TV)); };

    virtual size_type entity_type() const { return IFORM; };

    virtual size_type dof() const { return DOF; };
};

class AttributeDict : public concept::Printable
{
public:
    virtual std::ostream &print(std::ostream &os, int indent = 0) const;

    std::pair<std::shared_ptr<AttributeDesc>, bool>
    register_attr(std::shared_ptr<AttributeDesc> const &desc);

    void erase(id_type const &id);

    void erase(std::string const &id);

    std::shared_ptr<AttributeDesc> find(id_type const &id);

    std::shared_ptr<AttributeDesc> find(std::string const &id);

    std::shared_ptr<AttributeDesc> const &get(std::string const &k) const;

    std::shared_ptr<AttributeDesc> const &get(id_type k) const;

private:
    std::map<std::string, id_type> m_key_id_;
    std::map<id_type, std::shared_ptr<AttributeDesc> > m_map_;
};

class AttributeCollection : public design_pattern::Observable<void(Patch *)>
{
    typedef design_pattern::Observable<void(Patch *)> base_type;
public:
    AttributeCollection(std::shared_ptr<AttributeDict> const &p = nullptr);

    virtual  ~AttributeCollection();

    virtual void connect(Attribute *observer);

    virtual void disconnect(Attribute *observer);

    virtual void accept(Patch *p) { base_type::accept(p); }

    template<typename TF>
    void foreach(TF const &fun)
    {
        design_pattern::Observable<void(Patch *)>::foreach(
                [&](observer_type &obj) { fun(static_cast<Attribute *>(&obj)); });
    }

private:
    std::shared_ptr<AttributeDict> m_dict_;
};

struct Attribute :
        public concept::Printable,
        public concept::LifeControllable,
        public design_pattern::Observer<void(Patch *)>
{
public:
    SP_OBJECT_BASE(Attribute);

    Attribute(std::shared_ptr<DataBlock> const &d = nullptr, std::shared_ptr<AttributeDesc> const &desc = nullptr);

    Attribute(AttributeCollection *p, std::shared_ptr<AttributeDesc> const &desc);

    Attribute(Attribute const &other) = delete;

    Attribute(Attribute &&other) = delete;

    virtual ~Attribute();

    virtual std::ostream &print(std::ostream &os, int indent = 0) const { return os; };

    virtual std::shared_ptr<Attribute> clone() const =0;

//    virtual std::shared_ptr<DataBlock> create_data_block(void *p, std::shared_ptr<MeshBlock> const &m) const =0;

    virtual AttributeDesc const &desc() const { return *m_desc_; }

    virtual std::shared_ptr<DataBlock> const &data() const { return m_data_; }

    virtual std::shared_ptr<DataBlock> &data() { return m_data_; }

    virtual void pre_process();

    virtual void post_process();

    virtual void clear();

    virtual void accept(Patch *p);

    virtual void accept(std::shared_ptr<DataBlock> const &d);

private:
    std::shared_ptr<AttributeDesc> m_desc_ = nullptr;
    std::shared_ptr<DataBlock> m_data_;
};

template<typename TV, MeshEntityType IFORM, int DOF>
class DataAttribute : public Attribute
{
    typedef TV value_type;
    typedef DataAttribute<TV, IFORM, DOF> this_type;
    typedef DataBlockArray<TV, IFORM, DOF> data_entity_type;
public:
    template<typename ...Args>
    DataAttribute(Args &&...args):
            Attribute(nullptr, std::make_shared<AttributeDescTemp<TV, IFORM, DOF>>(std::forward<Args>(args)...)) {}

    ~DataAttribute() {}

    virtual std::shared_ptr<Attribute> clone() const { return std::make_shared<this_type>(); };

    virtual std::shared_ptr<DataBlock> create_data_block(void *p, std::shared_ptr<MeshBlock> const &m) const
    {
        return data_entity_type::create(m, static_cast<value_type *>(p));
    };

    template<typename ...Args>
    value_type &get(Args &&...args) { return m_data_->get(std::forward<Args>(args)...); }

    template<typename ...Args>
    value_type const &get(Args &&...args) const { return m_data_->get(std::forward<Args>(args)...); }

    virtual void pre_process()
    {
        Attribute::pre_process();
        m_data_ = static_cast<data_entity_type *>( Attribute::data().get());
    };

    virtual void post_process()
    {
        m_data_ = nullptr;
        Attribute::post_process();
    }

public:
    data_entity_type *m_data_;

};


}} //namespace data_block
#endif //SIMPLA_ATTRIBUTE_H

