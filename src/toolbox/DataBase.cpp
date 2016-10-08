//
// Created by salmon on 16-10-8.
//
#include "DataBase.h"

namespace simpla { namespace toolbox
{
std::ostream &DataBase::print(std::ostream &os, int indent) const
{
    size_type s = size();

    if (!value().is_null()) { value().print(os, indent); }
    else if (s == 1)
    {
        os << "{ ";

        for_each([&](std::string const &key, DataBase const &value)
                 {
                     os << key << " = ";
                     value.print(os, indent + 1);
                 });

        os << " }";
    } else
    {

        os << std::endl << std::setw(indent) << " " << "{ ";

        for_each([&](std::string const &key, DataBase const &value)
                 {
                     os << std::endl << std::setw(indent + 1) << "  " << key << " = ";
                     value.print(os, indent + 2);
                     os << ", ";
                 });

        os << std::endl << std::setw(indent) << " " << "}";
    }

    os << " ";
    return os;


};

std::ostream &operator<<(std::ostream &os, DataBase const &prop) { return prop.print(os, 0); }

}}
//namespace simpla{namespace toolbox{