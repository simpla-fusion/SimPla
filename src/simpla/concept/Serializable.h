//
// Created by salmon on 16-11-2.
//

#ifndef SIMPLA_SERIALIZABLE_H
#define SIMPLA_SERIALIZABLE_H

#include <string>

namespace simpla { namespace data { class DataEntityTable; }}

namespace simpla { namespace concept
{

struct Serializable
{
    Serializable() {}

    virtual ~Serializable() {}

    virtual std::string  name() const =0;

    virtual void load(data::DataEntityTable const &) =0;

    virtual void save(data::DataEntityTable *) const =0;
};


}}
#endif //SIMPLA_SERIALIZABLE_H
