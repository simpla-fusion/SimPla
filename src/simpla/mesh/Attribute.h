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

#include "MeshBlock.h"
#include "DataBlock.h"

namespace simpla { namespace mesh
{
class Patch;

class AttributeCollection;

struct Attribute :
        public concept::Printable,
        public concept::Configurable,
        public concept::LifeControllable,
        public design_pattern::Observer<void(Patch &)>
{
    explicit Attribute(AttributeCollection *);

    Attribute(Attribute const &other) = delete;

    Attribute(Attribute &&other) = delete;

    virtual ~Attribute();

    virtual std::ostream &print(std::ostream &os, int indent = 0) const
    {
        if (m_data_ != nullptr) { m_data_->print(os, indent); }
        return os;
    };

    virtual MeshEntityType entity_type() const =0;

    virtual std::type_info const &value_type_info() const =0;

    virtual size_type dof() const =0;

    virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(Attribute); };

    DataBlock *data_block();

    DataBlock const *data_block() const;

    template<typename U>
    U const *data_as() const
    {
        auto const *d = data_block();
        ASSERT(d != nullptr);
        ASSERT(d->is_a(typeid(U)));
        return static_cast<U const *>(d);
    }

    template<typename U>
    U *data_as()
    {
        auto *d = data_block();
        ASSERT(d != nullptr);
        ASSERT(d->is_a(typeid(U)));
        return static_cast<U *>(d);
    }

    virtual std::shared_ptr<DataBlock> create_data_block(std::shared_ptr<MeshBlock> const &m, void *p) const =0;

    virtual void move_to(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<DataBlock> const &d = nullptr);

    virtual void pre_process();

    virtual void post_process();

    virtual void clear();

    virtual void notify(Patch &p);

private:
    std::shared_ptr<MeshBlock> m_mesh_block_ = nullptr;
    std::shared_ptr<DataBlock> m_data_ = nullptr;

};


class AttributeCollection : public design_pattern::Observable<void(Patch &)>
{
    typedef design_pattern::Observable<void(Patch &)> base_type;
public:

    void move_to(Patch &p) { notify(p); }

private:
};


}} //namespace data_block
#endif //SIMPLA_ATTRIBUTE_H

