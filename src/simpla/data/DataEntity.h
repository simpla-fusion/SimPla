//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H

#include <simpla/toolbox/Any.h>
#include "DataSet.h"
#include "DataType.h"
#include "DataSpace.h"

namespace simpla { namespace data
{

struct DataEntity
{
public:
    DataEntity() {}

    virtual ~DataEntity() {}

//    virtual void swap(DataEntity &other)=0;

    virtual bool is_a(std::type_info const &t_id) const { return t_id == typeid(DataEntity); }

    virtual std::ostream &print(std::ostream &os, int indent = 0) const { return os; };

    virtual bool is_null() const =0;

    virtual bool empty() const =0;

    virtual bool is_heavy_data() const =0;

//    virtual DataSpace dataspace() const =0;
//
//    virtual DataType datatype() const =0;

    virtual void *data() =0;

    virtual void const *data() const =0;

};

struct DataEntityLight : public DataEntity, public toolbox::Any
{
    typedef toolbox::Any base_type;
public:
    DataEntityLight() {}

    virtual ~DataEntityLight() {}

    virtual void swap(DataEntityLight &other) { base_type::swap(other); };

    virtual bool is_a(std::type_info const &t_id) const
    {
        return t_id == typeid(DataEntityLight) || base_type::type() == t_id;
    }

    virtual std::ostream &print(std::ostream &os, int indent = 0) const { return base_type::print(os, indent); };

    virtual bool is_null() const { return base_type::is_null(); };

    virtual bool is_heavy_data() const { return false; };

//    virtual DataSpace dataspace() const {};
//
//    virtual DataType datatype() const { UNIMPLEMENTED; };

    virtual void *data() { return base_type::data(); }

    virtual void const *data() const { return base_type::data(); }

};

struct DataEntityHeavy : public DataEntity
{
    typedef DataEntity base_type;
public:
    DataEntityHeavy() {}

    DataEntityHeavy(std::shared_ptr<void> d, DataType const &, DataSpace const &) {};

    virtual ~DataEntityHeavy() {}

    virtual void swap(DataEntityHeavy &other) { UNIMPLEMENTED; };

    virtual bool is_a(std::type_info const &t_id) const
    {
        return t_id == typeid(DataEntityHeavy) || base_type::is_a(t_id);
    }

    virtual std::ostream &print(std::ostream &os, int indent = 1) const =0;

//    virtual DataSpace dataspace() const =0;
//
//    virtual DataType datatype() const =0;

    virtual bool is_heavy_data() const { return true; };

    virtual void set(std::shared_ptr<void> d, DataType const &, DataSpace const &) {};

    virtual void deploy() = 0;

    virtual void clear() = 0;

    virtual bool is_valid() const = 0;

    virtual bool empty() const = 0;

    virtual void *data() = 0;

    virtual void const *data() const = 0;
};


}}
#endif //SIMPLA_DATAENTITY_H
