//
// Created by salmon on 16-11-9.
//

#ifndef SIMPLA_CONFIGABLE_H
#define SIMPLA_CONFIGABLE_H

#include <>
namespace simpla { namespace data { class DataBase; }}

namespace simpla { namespace concept
{
struct Configable
{
    data::DataBase &db(std::string const &s = "") { return m_db_.get(s); }

    data::DataBase const &db(std::string const &s = "") const { return m_db_.at(s); }

private:
    data::DataBase m_db_;
};
}}
#endif //SIMPLA_CONFIGABLE_H
