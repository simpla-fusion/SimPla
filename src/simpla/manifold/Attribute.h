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
#include <simpla/manifold/Chart.h>
#include <simpla/manifold/Patch.h>
#include "DataBlock.h"

namespace simpla { namespace mesh
{

class AttributeCollection;

struct AttributeDesc : public concept::Configurable
{
    virtual std::type_index const &value_type_index() const =0;

    virtual MeshEntityType entity_type() const =0;

    virtual size_type dof() const =0;
};

struct Attribute :
        public concept::Printable,
        public concept::LifeControllable,
        public design_pattern::Observer<void(std::shared_ptr<Patch> const &)>
{
public:
    SP_OBJECT_BASE(Attribute);

    explicit Attribute(AttributeCollection *);

    Attribute(Attribute const &other) = delete;

    Attribute(Attribute &&other) = delete;

    virtual ~Attribute();

    virtual std::ostream &print(std::ostream &os, int indent = 0) const { return os; };

    virtual MeshEntityType entity_type() const =0;

    virtual std::type_info const &value_type_info() const =0;

    virtual size_type dof() const =0;

    virtual std::shared_ptr<Attribute> clone() const =0;

    virtual std::shared_ptr<DataBlock> create_data_block(std::shared_ptr<Chart> const &m, void *p) const =0;

    virtual AttributeDesc const &desc() { return *m_desc_; }

    template<typename U> U const *data_as() const { return m_data_->as<U>(); }

    template<typename U> U *data_as() { return m_data_->as<U>(); }

    template<typename U> U const *mesh_as() const { return m_mesh_->as<U>(); }


    virtual void move_to(std::shared_ptr<Chart> const &m, std::shared_ptr<DataBlock> const &d = nullptr);

    virtual void pre_process();

    virtual void post_process();

    virtual void clear();

    virtual void notify(std::shared_ptr<Patch> const &p);

private:
    std::shared_ptr<AttributeDesc> m_desc_ = nullptr;
    std::shared_ptr<Chart> m_mesh_;
    std::shared_ptr<DataBlock> m_data_;
};


class AttributeCollection : public design_pattern::Observable<void(std::shared_ptr<Patch> const &)>
{
    typedef design_pattern::Observable<void(std::shared_ptr<Patch> const &)> base_type;
public:

    void move_to(std::shared_ptr<Patch> const &p) { notify(p); }

private:
};


}} //namespace data_block
#endif //SIMPLA_ATTRIBUTE_H

