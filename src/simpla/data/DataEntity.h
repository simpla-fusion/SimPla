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

struct DataEntity : public toolbox::Any
{
    typedef toolbox::Any base_type;
public:
    template<typename ...Args>
    DataEntity(Args &&...args) : base_type(std::forward<Args>(args)...) {}

    DataEntity(DataEntity const &other) : base_type(other) {}

    DataEntity(DataEntity &&other) : base_type(other) {}

    virtual ~DataEntity() {}

    void swap(DataEntity &other) { base_type::swap(other); }

    DataEntity &operator=(const DataEntity &rhs)
    {
        DataEntity(rhs).swap(*this);
        return *this;
    }

    // move assignement
    DataEntity &operator=(DataEntity &&rhs)
    {
        rhs.swap(*this);
        DataEntity().swap(rhs);
        return *this;
    }


    virtual bool is_a(std::type_info const &t_id) const { return t_id == typeid(DataEntity); }

    virtual const std::type_info &value_type_info() const { return toolbox::Any::type(); }

    virtual bool is_heavy_data() const { return false; }

    template<typename U>
    DataEntity &operator=(U const &v)
    {
        base_type::operator=(v);
        return *this;
    };

};

struct DataEntityLight : public DataEntity
{
};

struct DataEntityHeavy : public DataEntityLight
{
    typedef DataEntity base_type;
public:
    DataEntityHeavy() {}

    DataEntityHeavy(std::shared_ptr<void> d, DataType const &, DataSpace const &) {};

    virtual ~DataEntityHeavy() {}

    virtual bool is_heavy_data() const { return true; };

    virtual bool is_a(std::type_info const &t_id) const
    {
        return t_id == typeid(DataEntityHeavy) || base_type::is_a(t_id);
    }

    virtual std::ostream &print(std::ostream &os, int indent = 1) const { return os; };

//    virtual DataSpace dataspace() const =0;
//
//    virtual DataType datatype() const =0;


//    virtual void set(std::shared_ptr<void> d, DataType const &, DataSpace const &) {};
//
//    virtual void deploy() = 0;
//
//    virtual void clear() = 0;
//
//    virtual bool is_valid() const = 0;
//
//    virtual bool empty() const = 0;
//
//    virtual void *data() = 0;
//
//    virtual void const *data() const = 0;



};


}}
#endif //SIMPLA_DATAENTITY_H
