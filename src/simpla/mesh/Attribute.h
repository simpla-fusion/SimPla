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

#include "MeshBlock.h"
#include "DataBlock.h"

namespace simpla { namespace mesh
{
class Chart;

class AttributeCollection;

/**
 *  AttributeBase IS-A container of data blocks
 *  Define of attribute
 *  *  is printable
 *  *  is serializable
 *  *  is unaware of mesh block
 *  *  is unaware of mesh atlas
 *  *  has a value type
 *  *  has a MeshEntityType (entity_type)
 *  *  has n data block (emplace, erase,find,at)
 *  *
 */
class AttributeBase :
        public Object,
        public concept::Printable,
        public concept::Serializable,
        public concept::Configurable,
        public std::enable_shared_from_this<AttributeBase>
{


public:

    SP_OBJECT_HEAD(AttributeBase, Object)

    AttributeBase(std::string const &config_str = "");

    AttributeBase(AttributeBase const &) = delete;

    AttributeBase(AttributeBase &&) = delete;

    virtual ~AttributeBase();

    virtual MeshEntityType entity_type() const =0;

    virtual std::type_info const &value_type_info() const =0;

    virtual size_t dof() const { return 1; };

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual void load(const data::DataEntityTable &);

    virtual void save(data::DataEntityTable *) const;

    virtual const DataBlock *find(const id_type &) const;

    virtual DataBlock *find(const id_type &);

    virtual void erase(const id_type &);

    virtual std::shared_ptr<DataBlock> const &at(const id_type &m) const;

    virtual std::shared_ptr<DataBlock> &at(const id_type &m);

    virtual std::pair<std::shared_ptr<DataBlock>, bool> emplace(const id_type &m, const std::shared_ptr<DataBlock> &p);

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

struct AttributeView;

template<typename TV, MeshEntityType IFORM, size_type IDOF>
class Attribute : public AttributeBase
{
public:
    static constexpr mesh::MeshEntityType iform = IFORM;
    static constexpr size_type DOF = IDOF;

    template<typename ...Args>
    explicit Attribute(Args &&...args):AttributeBase(std::forward<Args>(args)...) {}

    virtual ~Attribute() {}

    MeshEntityType entity_type() const { return IFORM; };

    std::type_info const &value_type_info() const { return typeid(typename traits::value_type<TV>::type); };

    size_type dof() const { return DOF; };

    std::shared_ptr<AttributeView>
    view(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<DataBlock> const &p = nullptr)
    {
        auto res = AttributeBase::emplace(m->id(), p);
        return std::make_shared<AttributeView>(m, res.first, this->shared_from_this());
    };


};

class AttributeViewCollection;

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


struct AttributeView :
        public Object,
        public concept::Printable,
        public concept::LifeControllable
{
    SP_OBJECT_HEAD(AttributeView, Object);
public:
    explicit AttributeView(std::shared_ptr<AttributeBase> const &attr = nullptr);

    explicit AttributeView(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<DataBlock> const &d,
                           std::shared_ptr<AttributeBase> const &attr = nullptr);

    AttributeView(AttributeView const &other) = delete;

    AttributeView(AttributeView &&other) = delete;

    virtual ~AttributeView();

    virtual std::ostream &print(std::ostream &os, int indent = 0) const { return os; };

    virtual MeshEntityType entity_type() const { return m_attr_->entity_type(); };

    virtual std::type_info const &value_type_info() const { return m_attr_->value_type_info(); };

    virtual size_type dof() const { return m_attr_->dof(); };

    std::shared_ptr<AttributeBase> &attribute() { return m_attr_; }

    std::shared_ptr<AttributeBase> const &attribute() const { return m_attr_; }

    DataBlock *data() { return m_data_block_; }

    DataBlock const *data() const { return m_data_block_; }

    void move_to(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<DataBlock> const &d);

    virtual std::shared_ptr<DataBlock> create_data_block(MeshBlock const *m, void *p) const =0;

    virtual void deploy();

    virtual void pre_process();

    virtual void initialize();

    virtual void post_process();

    virtual void destroy();

    template<typename U> U *
    data_as()
    {
        auto *res = data();
        return (res == nullptr) ? nullptr : res->as<U>();
    }

    template<typename U> U const *
    data_as() const
    {
        auto *res = data();
        return (res == nullptr) ? nullptr : res->as<U>();
    }


private:
    std::shared_ptr<AttributeBase> m_attr_ = nullptr;
    std::shared_ptr<DataBlock> m_data_block_holder_ = nullptr;
    std::shared_ptr<MeshBlock> m_mesh_block_holder_ = nullptr;
    DataBlock *m_data_block_;
    MeshBlock const *m_mesh_block_;
};

class AttributeViewCollection
{
public:


    void disconnect(AttributeView *v) { m_views_.insert(v); };

    void connect(AttributeView *v) { m_views_.erase(v); };

    std::set<AttributeView *> &attributes() { return m_views_; }

    std::set<AttributeView *> const &attributes() const { return m_views_; }

private:
    std::set<AttributeView *> m_views_;

};

class AttributeCollection :
        public concept::Printable,
        public concept::Serializable,
        public concept::Configurable
{
    typedef id_type key_type;
public:
    AttributeCollection();

    virtual ~AttributeCollection();

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual void load(const data::DataEntityTable &);

    virtual void save(data::DataEntityTable *) const;

    const AttributeBase *find(const key_type &) const;

    AttributeBase *find(const key_type &);

    void erase(const key_type &);

    std::shared_ptr<AttributeBase> const &at(const key_type &m) const;

    std::shared_ptr<AttributeBase> &at(const key_type &m);

    std::pair<std::shared_ptr<AttributeBase>, bool>
    emplace(const key_type &k, const std::shared_ptr<AttributeBase> &p);


private:
    std::map<key_type, std::shared_ptr<AttributeBase>> m_map_;

};

}} //namespace data_block
#endif //SIMPLA_ATTRIBUTE_H

