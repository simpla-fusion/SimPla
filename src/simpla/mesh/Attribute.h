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
        public concept::LifeControllable
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

    virtual std::shared_ptr<DataBlock> create_data_block(MeshBlock const *m, void *p) const =0;

    DataBlock *data() { return m_data_block_; }

    DataBlock const *data() const { return m_data_block_; }

    void move_to(std::shared_ptr<MeshBlock> const &m, std::shared_ptr<DataBlock> const &d)
    {
        if (m == nullptr || (m == m_mesh_block_holder_ && d == m_data_block_holder_)) { return; }

        post_process();

        m_mesh_block_holder_ = m;
        m_data_block_holder_ = d;
    }


    virtual void deploy() {}

    virtual void pre_process()
    {
        if (is_valid()) { return; } else { concept::LifeControllable::pre_process(); }

        if (m_mesh_block_holder_ != nullptr) { m_mesh_block_ = m_mesh_block_holder_.get(); }

        ASSERT(m_mesh_block_ != nullptr);

        if (m_data_block_holder_ != nullptr) { m_data_block_ = m_data_block_holder_.get(); }

        if (m_data_block_ == nullptr)
        {
            m_data_block_holder_ = create_data_block(m_mesh_block_, nullptr);
            m_data_block_ = m_data_block_holder_.get();
        }
        ASSERT(m_data_block_ != nullptr);

        m_data_block_->pre_process();


    }

    virtual void initialize()
    {
        pre_process();
        auto *p = data();
        if (p != nullptr) p->clear();
    }

    virtual void post_process()
    {
        if (!is_valid()) { return; } else { concept::LifeControllable::post_process(); }
        m_data_block_ = nullptr;
        m_mesh_block_ = nullptr;
        m_data_block_holder_.reset();
        m_mesh_block_holder_.reset();
    }

    virtual void destroy() {};

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
    std::shared_ptr<DataBlock> m_data_block_holder_ = nullptr;
    std::shared_ptr<MeshBlock> m_mesh_block_holder_ = nullptr;
    DataBlock *m_data_block_;
    MeshBlock const *m_mesh_block_;
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

