//
// Created by salmon on 16-10-28.
//

#ifndef SIMPLA_DATAENTITY_H
#define SIMPLA_DATAENTITY_H

#include "Any.h"
#include "DataSet.h"

namespace simpla { namespace toolbox
{

struct DataEntity : public Any
{
public:
    DataEntity() {}

    virtual ~DataEntity() {}

    virtual bool is_a(std::type_info const &t_id) const { return Any::type() == typeid(DataEntity); }

    virtual std::ostream &print(std::ostream &os, int indent = 0) const { return Any::print(os, indent); };

    virtual bool is_null() const { return Any::is_null(); };

    virtual bool is_boolean() const { return Any::is_boolean(); }

    virtual bool is_integral() const { return Any::is_integral(); }

    virtual bool is_floating_point() const { return Any::is_floating_point(); }

    virtual bool is_string() const { return Any::is_string(); }

    virtual bool is_heavy_data() const { return false; };


//    std::shared_ptr<DataSet> dataset() { return nullptr; };
//
//    std::shared_ptr<DataSet> dataset() const { return nullptr; };
};

}}
#endif //SIMPLA_DATAENTITY_H
