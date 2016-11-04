//
// Created by salmon on 16-10-20.
//

#ifndef SIMPLA_ATTRIBUTE_H
#define SIMPLA_ATTRIBUTE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/Serializable.h>
#include <simpla/toolbox/Printable.h>

#include "MeshBlock.h"
#include "DataBlock.h"

namespace simpla { namespace mesh
{

/**
 *  AttributeBase IS-A container of datablock
 */
class AttributeBase :
        public toolbox::Object, public toolbox::Serializable,
        public std::enable_shared_from_this<AttributeBase>
{


public:

    SP_OBJECT_HEAD(AttributeBase, toolbox::Object)

    AttributeBase(std::string const &s);

    AttributeBase(AttributeBase const &) = delete;

    AttributeBase(AttributeBase &&) = delete;

    virtual ~AttributeBase();


    virtual std::type_info const &value_type_info() const =0;

    virtual mesh::MeshEntityType entity_type() const =0;

    virtual std::string const &name() const { return toolbox::Object::name(); };

    virtual std::ostream &print(std::ostream &os, int indent = 1) const;

    virtual void load(const data::DataBase &);

    virtual void save(data::DataBase *) const;

    void insert(id_type id, const std::shared_ptr<DataBlockBase> &);

//    virtual DataBlockBase &create(MeshBlock const &)=0;

    virtual DataBlockBase &create(const MeshBlock *, id_type hint);

    virtual bool has(const id_type &t_id) const;

    virtual void erase(id_type id);

    virtual void deploy(id_type id = 0);

    virtual void clear(id_type id = 0);

    virtual void update(id_type dest, id_type src = 0);

    virtual DataBlockBase &data(id_type t_id = 0);

    virtual DataBlockBase const &data(id_type t_id = 0) const;


private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;


};

template<typename TV, MeshEntityType IFORM>
class Attribute : public AttributeBase
{
public:
    Attribute(std::string const &s) : AttributeBase(s) {}

    virtual ~Attribute() {}

    typedef TV value_type;

    typedef DataBlock<value_type, IFORM> data_block_type;

    static std::shared_ptr<Attribute<TV, IFORM>> create() { return std::make_shared<Attribute<TV, IFORM>>(); };

    virtual std::type_info const &value_type_info() const { return typeid(value_type); };

    virtual mesh::MeshEntityType entity_type() const { return IFORM; }

    virtual data_block_type &
    data(id_type id = 0) { return static_cast<data_block_type &>(AttributeBase::data(id)); };

    virtual data_block_type const &
    data(id_type id = 0) const { return static_cast<data_block_type const &>(AttributeBase::data(id)); };
};


}} //namespace data
#endif //SIMPLA_ATTRIBUTE_H

