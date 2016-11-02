//
// Created by salmon on 16-11-2.
//

#ifndef SIMPLA_SERIALIZABLE_H
#define SIMPLA_SERIALIZABLE_H

#include <string>

namespace simpla { namespace data
{
class DataBase;

struct Serializable
{
    Serializable() {}

    virtual ~Serializable() {}

    virtual std::string const &name() const =0;

    virtual void load(DataBase const &) =0;

    virtual void save(DataBase *) const =0;
};


}}
#endif //SIMPLA_SERIALIZABLE_H
