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
#include <simpla/toolbox/design_pattern/Observer.h>

#include "DataBlock.h"

namespace simpla { namespace mesh
{
class Patch;


struct AttributeDesc : public concept::Configurable, public Object
{
    virtual std::type_index const &value_type_index() const =0;

    virtual MeshEntityType entity_type() const =0;

    virtual size_type dof() const =0;
};

struct Attribute :
        public concept::Printable,
        public concept::LifeControllable,
        public design_pattern::Observer<void(Patch *)>
{
public:
    SP_OBJECT_BASE(Attribute);

    Attribute();

    template<typename ...Args>
    explicit Attribute(AttributeCollection *p, Args &&...args)
            :Attribute(p),
             m_desc_(std::make_shared<AttributeDesc>(std::forward<Args>(args)...)) {};

    explicit Attribute(AttributeCollection *p);

    Attribute(Attribute const &other) = delete;

    Attribute(Attribute &&other) = delete;

    virtual ~Attribute();

    virtual std::ostream &print(std::ostream &os, int indent = 0) const { return os; };

    virtual MeshEntityType entity_type() const =0;

    virtual std::type_info const &value_type_info() const =0;

    virtual size_type dof() const =0;

    virtual std::shared_ptr<Attribute> clone() const =0;

    virtual std::shared_ptr<DataBlock> create_data_block(std::shared_ptr<MeshBlock> const &m, void *p) const =0;

    virtual AttributeDesc const &desc() { return *m_desc_; }

    template<typename U> U const *data_as() const { return m_data_->as<U>(); }

    template<typename U> U *data_as() { return m_data_->as<U>(); }

    template<typename U> U const *mesh_as() const { return m_mesh_->as<U>(); }

    std::shared_ptr<MeshBlock> mesh() const { return m_mesh_; }


    virtual void pre_process();

    virtual void post_process();

    virtual void clear();

    virtual void accept(Patch *p);

    virtual void accept(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<DataBlock> const &d = nullptr);

private:
    std::shared_ptr<AttributeDesc> m_desc_ = nullptr;
    std::shared_ptr<MeshBlock> m_mesh_;
    std::shared_ptr<DataBlock> m_data_;
};

template<typename TV, MeshEntityType IFORM, int DOF>
class DataAttribute : public Attribute
{
    typedef TV value_type;
    typedef data::DataEntityNDArray<TV> data_entity_type;
public:
    template<typename ...Args>
    DataAttribute(Args &&...args):    Attribute(std::forward<Args>(args)...) {}

    ~DataAttribute() {}

    template<typename ...Args>
    value_type &get(Args &&...args) { return m_data_->get(std::forward<Args>(args)...); }

    template<typename ...Args>
    value_type const &get(Args &&...args) const { return m_data_->get(std::forward<Args>(args)...); }

public:
    data_entity_type *m_data_;

};

class AttributeCollection : public design_pattern::Observable<void(Patch *)>
{
    typedef design_pattern::Observable<void(Patch *)> base_type;
public:

    virtual void accept(Patch *p) { base_type::accept(p); }

private:
};


}} //namespace data_block
#endif //SIMPLA_ATTRIBUTE_H

