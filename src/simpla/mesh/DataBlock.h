//
// Created by salmon on 16-11-2.
//

#ifndef SIMPLA_PATCH_H
#define SIMPLA_PATCH_H

#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/PrettyStream.h>
#include <simpla/concept/Serializable.h>
#include <simpla/concept/Printable.h>
#include <simpla/data/DataEntityNDArray.h>

namespace simpla { namespace mesh
{
/**
 *  Base class of Data Blocks (pure virtual)
 */

struct DataBlock : public concept::Serializable, public concept::Printable
{
public:
    DataBlock() {}

    virtual ~DataBlock() {}

    virtual bool is_a(std::type_info const &info) const { return info == typeid(DataBlock); };

    virtual std::type_info const &value_type_info() const =0;

    virtual mesh::MeshEntityType entity_type() const =0;

    virtual std::string name() const =0;

    virtual void load(data::DataBase const &) =0;

    virtual void save(data::DataBase *) const =0;

    virtual std::ostream &print(std::ostream &os, int indent) const =0;

    virtual std::shared_ptr<DataBlock> clone(MeshBlock const *) const =0;

    virtual void sync(std::shared_ptr<DataBlock>, bool only_ghost = true)   =0;

    virtual bool is_deployed() const =0;

    virtual void deploy() =0;

    virtual void destroy() =0;

};

template<typename TV, MeshEntityType IFORM>
class DataBlockArray : public DataBlock, public data::DataEntityNDArray<TV, 4>
{
    typedef DataBlockArray<TV, IFORM> this_type;
    typedef data::DataEntityNDArray<TV, 4> data_entity_type;
public:
    typedef TV value_type;

    DataBlockArray() : DataBlock(), data_entity_type() {}

    template<typename ...Args>
    DataBlockArray(Args &&...args) :
            DataBlock(),
            data_entity_type(std::forward<Args>(args)...) {}

    virtual ~DataBlockArray() {}

    virtual std::type_info const &value_type_info() const { return typeid(value_type); };

    virtual mesh::MeshEntityType entity_type() const { return IFORM; }

    virtual bool is_a(std::type_info const &info) const
    {
        return info == typeid(DataBlockArray<TV, IFORM>) || DataBlock::is_a(info);
    };

    virtual std::string name() const { return ""; }

    virtual void load(data::DataBase const &) {};

    virtual void save(data::DataBase *) const {};

    virtual std::ostream &print(std::ostream &os, int indent) const
    {
        os << " type = \'" << value_type_info().name() << "\' "
           << ", entity type = " << static_cast<int>(entity_type())
           << ", data = {";
        data_entity_type::print(os, indent + 1);
        os << "}";
        return os;
    }


    virtual std::shared_ptr<DataBlock> clone(MeshBlock const *m) const
    {
        // FIXME :: data block is not initializied!!
        return std::dynamic_pointer_cast<DataBlock>(std::make_shared<this_type>());
    };

    virtual bool is_deployed() const { return data_entity_type::is_deployed(); };

    virtual void deploy() { data_entity_type::deploy(); };

    virtual void destroy() { data_entity_type::destroy(); };

    virtual void sync(std::shared_ptr<DataBlock>, bool only_ghost = true) { UNIMPLEMENTED; };

};
}}//namespace simpla { namespace mesh

#endif //SIMPLA_PATCH_H
