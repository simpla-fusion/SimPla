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


/**
 *  AttributeBase IS-A container of data blocks
 *  Define of attribute
 *  *  is printable
 *  *  is serializable
 *  *  is unware of mesh block
 *  *  is unware of mesh atlas
 *  *  has a value type
 *  *  has a MeshEntityType (entity_type)
 *  *  has n data block (insert, erase,has,at)
 *  *
 */
class AttributeBase :
        public Object,
        public concept::Printable,
        public concept::Serializable,
        public concept::Configurable
{


public:

    SP_OBJECT_HEAD(AttributeBase, Object)


    AttributeBase(std::string const &s = "", std::string const &config_str = "");

    AttributeBase(AttributeBase const &) = delete;

    AttributeBase(AttributeBase &&) = delete;

    virtual ~AttributeBase();

    virtual MeshEntityType entity_type() const =0;

    virtual std::type_info const &value_type_info() const =0;

    virtual size_t dof() const { return 1; };

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual std::string name() const { return m_name_; };

    virtual void load(const data::DataBase &);

    virtual void save(data::DataBase *) const;

    virtual bool has(const id_type &) const;

    virtual void erase(const id_type &);

    virtual std::shared_ptr<DataBlock> const &at(const id_type &m) const;

    virtual std::shared_ptr<DataBlock> &at(const id_type &m);

    virtual std::shared_ptr<DataBlock> insert_or_assign(const id_type &m, const std::shared_ptr<DataBlock> &p);


private:
    std::string m_name_;
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

template<typename TV, MeshEntityType IFORM, size_type IDOF>
class Attribute : public AttributeBase
{
public:
    static constexpr mesh::MeshEntityType iform = IFORM;
    static constexpr size_type DOF = IDOF;

    template<typename ...Args>
    explicit Attribute(Args &&...args):AttributeBase(std::forward<Args>(args)...) {}

    virtual ~Attribute() {}

    virtual MeshEntityType entity_type() const { return IFORM; };

    virtual std::type_info const &value_type_info() const { return typeid(typename traits::value_type<TV>::type); };

    virtual size_type dof() const { return DOF; };

};


struct AttributeView : public concept::Printable, public concept::Deployable
{
    explicit AttributeView(std::shared_ptr<AttributeBase> const &attr = nullptr);

    template<typename U>
    explicit AttributeView(std::shared_ptr<U> const &attr):
            AttributeView(std::dynamic_pointer_cast<AttributeBase>(attr)) {};

    AttributeView(AttributeView const &other) = delete;

    AttributeView(AttributeView &&other) = delete;

    virtual ~AttributeView();

    virtual std::ostream &print(std::ostream &os, int indent = 0) const
    {
        if (m_data_ != nullptr) { m_data_->print(os, indent); }
        return os;
    };

    virtual MeshEntityType entity_type() const =0;

    virtual std::type_info const &value_type_info() const =0;

    virtual size_type dof() const =0;

    virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(AttributeView); };

    id_type mesh_id() const;

    std::shared_ptr<AttributeBase> &attribute();

    std::shared_ptr<AttributeBase> const &attribute() const;

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

    virtual void preprocess();

    virtual void postprocess();

    virtual void clear();


private:
    std::shared_ptr<MeshBlock> m_mesh_block_ = nullptr;
    std::shared_ptr<DataBlock> m_data_ = nullptr;
    std::shared_ptr<AttributeBase> m_attr_ = nullptr;

};


/**
 * AttributeView: expose one block of attribute
 * -) is a view of Attribute
 * -) is unaware of the type of Mesh
 * -) has a pointer to a mesh block
 * -) has a pointer to a data block
 * -) has a shared pointer of attribute
 * -) can traverse on the Attribute
 * -) if there is no Atlas, AttributeView will hold the MeshBlock
 * -) if there is no AttributeHolder, AttributeView will hold the DataBlock and Attribute
 */



}} //namespace data_block
#endif //SIMPLA_ATTRIBUTE_H

