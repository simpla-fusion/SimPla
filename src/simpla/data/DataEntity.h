//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H

#include <simpla/toolbox/Any.h>
#include "DataSet.h"

namespace simpla { namespace toolbox
{

struct DataEntity : public Any
{
public:
    DataEntity() {}

    virtual ~DataEntity() {}

    virtual void swap(DataEntity &other) { Any::swap(other); };

    virtual bool is_a(std::type_info const &t_id) const { return Any::type() == t_id || t_id == typeid(DataEntity); }

    virtual std::ostream &print(std::ostream &os, int indent = 0) const { return Any::print(os, indent); };

    virtual bool is_null() const { return Any::is_null(); };

    virtual bool is_boolean() const { return Any::is_boolean(); }

    virtual bool is_integral() const { return Any::is_integral(); }

    virtual bool is_floating_point() const { return Any::is_floating_point(); }

    virtual bool is_string() const { return Any::is_string(); }

    virtual bool is_heavy_data() const { return false; };
};

struct HeavyDataEntity : public DataEntity
{
    HeavyDataEntity() {}

    HeavyDataEntity(std::shared_ptr<void> d, DataType const &, DataSpace const &) {};

    virtual bool is_a(std::type_info const &t_id) const { return t_id == typeid(HeavyDataEntity); }

    virtual ~HeavyDataEntity() {}

    void swap(DataEntity &other) { UNIMPLEMENTED; };

    void swap(HeavyDataEntity &other) { UNIMPLEMENTED; };

    virtual bool is_heavy_data() const { return true; };

    virtual std::shared_ptr<void> data() const { return nullptr; };

    virtual DataSpace dataspace() {};

    virtual DataType datatype() {};

    virtual void set(std::shared_ptr<void> d, DataType const &, DataSpace const &) {};
};
}}
#endif //SIMPLA_DATAENTITY_H
