//
// Created by salmon on 16-11-17.
//

#ifndef SIMPLA_CONFIGURABLE_H
#define SIMPLA_CONFIGURABLE_H

#include <simpla/data/DataEntityTable.h>

namespace simpla { namespace concept
{

struct Configurable
{
    data::DataEntityTable db;

    Configurable() {}

    virtual ~Configurable() {}

    std::string name() const { return db.get_value("name", std::string("")); }

};


}}
#endif //SIMPLA_CONFIGURABLE_H
