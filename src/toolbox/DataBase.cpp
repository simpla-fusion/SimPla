//
// Created by salmon on 16-10-8.
//
#include "DataBase.h"

namespace simpla { namespace toolbox
{

std::ostream &operator<<(std::ostream &os, DataEntity const &prop) { return prop.print(os, 0); }

std::ostream &operator<<(std::ostream &os, DataBase const &prop) { return prop.print(os, 0); }

std::ostream &DataBase::print(std::ostream &os, int indent) const
{

    if (size() == 1)
    {
        os << "{ ";

        foreach([&](std::string const &key, DataEntity const &value)
                {
                    os << key << " = ";
                    value.print(os, indent + 1);
                });

        os << " }";
    } else
    {

        os << std::endl << std::setw(indent) << " " << "{ ";

        foreach([&](std::string const &key, DataEntity const &value)
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


}}
//namespace simpla{namespace toolbox{