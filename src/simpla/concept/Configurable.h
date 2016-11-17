//
// Created by salmon on 16-11-17.
//

#ifndef SIMPLA_CONFIGURABLE_H
#define SIMPLA_CONFIGURABLE_H

#include <simpla/data/DataBase.h>

namespace simpla { namespace concept
{

struct Configurable
{
    data::DataBase db;

    Configurable() {}

    virtual ~Configurable() {}

    data::DataBase &config(std::string const &s = "") { return db.get(s); }

    data::DataBase const &config(std::string const &s = "") const { return db.at(s); }


};


}}
#endif //SIMPLA_CONFIGURABLE_H
