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

    data::DataEntityLight &config(std::string const &s = "") { return db.get(s)->as_light(); }

    data::DataEntityLight const &config(std::string const &s = "") const { return db.at(s).as_light(); }


};


}}
#endif //SIMPLA_CONFIGURABLE_H
