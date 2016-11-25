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


struct AttributeViewBase : public concept::Printable
{

    AttributeViewBase(std::shared_ptr<AttributeBase> const &attr = nullptr);

    AttributeViewBase(AttributeViewBase const &other);

    AttributeViewBase(AttributeViewBase &&other);

    virtual void swap(AttributeViewBase &other);

    virtual ~AttributeViewBase();


    virtual MeshEntityType entity_type() const =0;

    virtual std::type_info const &value_type_info() const =0;

    virtual size_type dof() const =0;

    virtual bool is_a(std::type_info const &t_info) const =0;

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

    virtual void move_to(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<DataBlock> const &d = nullptr);

    virtual void deploy();

    virtual void clear();

    virtual void destroy();

    virtual std::shared_ptr<DataBlock> create_data_block(std::shared_ptr<MeshBlock> const &m)=0;

    virtual std::ostream &print(std::ostream &os, int indent = 0) const
    {
        if (m_data_ != nullptr) { m_data_->print(os, indent); }
        return os;
    };

private:
    id_type m_id_ = 0;
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
template<typename TV, MeshEntityType IFORM, size_type IDOF = 1>
class AttributeView : public AttributeViewBase
{

protected:

    typedef AttributeView this_type;
    typedef Attribute<TV, IFORM, IDOF> attribute_type;

public:

    static constexpr MeshEntityType iform = IFORM;

    static constexpr size_type DOF = IDOF;

    AttributeView() : AttributeViewBase(nullptr) {};

    template<typename ...Args>
    explicit AttributeView(Args &&...args) :
            AttributeViewBase(std::make_shared<attribute_type>(std::forward<Args>(args)...)) {};


    virtual ~AttributeView() {}

    AttributeView(this_type const &other) : AttributeViewBase(other) {};

    AttributeView(this_type &&other) : AttributeViewBase(std::forward<this_type>(other)) {};

    virtual void swap(this_type &other)
    {
        AttributeViewBase::swap(other);
    };


    virtual std::shared_ptr<DataBlock> create_data_block(std::shared_ptr<MeshBlock> const &m)
    {
        return m->create_data_block<TV, IFORM, DOF>();
    };

    virtual MeshEntityType entity_type() const { return IFORM; };

    virtual std::type_info const &value_type_info() const { return typeid(TV); };

    virtual size_type dof() const { return DOF; };

    virtual bool is_a(std::type_info const &t_info) const { return t_info == typeid(this_type); }
};

//    virtual void register_data_block_factory(std::shared_ptr<AttributeBase> const &attr) const
//    {
//        attr->register_data_block_factory(
//                std::type_index(typeid(MeshBlock)),
//                [&](const std::shared_ptr<MeshBlock> &m, void *p)
//                {
//                    ASSERT(p != nullptr);
//                    return m->template create_data_block<TV, IFORM, DOF>(p);
//                });
//    }

}} //namespace data_block
#endif //SIMPLA_ATTRIBUTE_H

