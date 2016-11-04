//
// Created by salmon on 16-11-2.
//

#ifndef SIMPLA_PATCH_H
#define SIMPLA_PATCH_H

#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/PrettyStream.h>
#include <simpla/toolbox/Serializable.h>
#include <simpla/toolbox/Printable.h>
#include "MeshBlock.h"

namespace simpla { namespace mesh
{
/**
 *  Base class of Data Blocks (pure virtual)
 */

struct DataBlock : public toolbox::Serializable, public toolbox::Printable
{
public:
    DataBlock(MeshBlock const *m) : m_(m) {}

    virtual ~DataBlock() {}

    virtual bool is_a(std::type_info const &info) const { return info == typeid(DataBlock); };

    virtual std::type_info const &value_type_info() const =0;

    virtual mesh::MeshEntityType entity_type() const =0;

    virtual std::string const &name() const =0;

    virtual void load(data::DataBase const &) =0;

    virtual void save(data::DataBase *) const =0;

    virtual std::ostream &print(std::ostream &os, int indent) const =0;

    virtual std::shared_ptr<DataBlock> create(MeshBlock const *) const =0;

    virtual void deploy() =0;

    virtual void clear() =0;

    virtual MeshBlock const *mesh() const { return m_; };

private:

    MeshBlock const *m_;

};

template<typename TV, MeshEntityType IFORM>
class DataBlockArray : public DataBlock
{
    typedef DataBlockArray<TV, IFORM> this_type;
public:
    typedef TV value_type;

    DataBlockArray(MeshBlock const *m) : DataBlock(m) {}

    virtual ~DataBlockArray() {}

    virtual std::type_info const &value_type_info() const { return typeid(value_type); };

    virtual mesh::MeshEntityType entity_type() const { return IFORM; }

    virtual bool is_a(std::type_info const &info) const
    {
        return info == typeid(DataBlockArray<TV, IFORM>) || DataBlock::is_a(info);
    };

    virtual std::string const &name() const { return ""; }

    virtual void load(data::DataBase const &) {};

    virtual void save(data::DataBase *) const {};

    virtual std::ostream &print(std::ostream &os, int indent) const
    {
        os << " type = \'" << value_type_info().name() << "\' "
           << ", entity type = " << static_cast<int>(entity_type())
           << ", level = " << mesh()->level();

        os << ", dimensions = " << mesh()->dimensions();

        return os;
    }


    virtual std::shared_ptr<DataBlock> create(MeshBlock const *m) const
    {
        return std::dynamic_pointer_cast<DataBlock>(std::make_shared<this_type>(m));
    };

    virtual void deploy() {};

    virtual void clear() {};
};
}}//namespace simpla { namespace mesh

#endif //SIMPLA_PATCH_H
