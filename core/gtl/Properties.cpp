/**
 * @file Properties.cpp
 * @author salmon
 * @date 2015-12-23.
 */

#include "Properties.h"

namespace simpla
{
std::ostream &Properties::print(std::ostream &os, int indent) const
{

    if (!this->any::empty()) { this->any::print(os, indent); }


    if (this->size() == 1)
    {
        auto it = this->begin();

        os << "{" << it->first << " = ";

        it->second.print(os, indent + 1);

        os << "}";
    }
    else if (this->size() > 1)
    {
        auto it = this->begin();
        auto ie = this->end();


        os << std::endl << std::setw(indent + 1) << "{" << std::endl;


        os << std::setw(indent + 1) << " " << it->first << " = ";
        it->second.print(os, indent + 1);
        ++it;

        for (; it != ie; ++it)
        {
            os << " , " << std::endl
            << std::setw(indent + 1) << " " << it->first << " = ";
            it->second.print(os, indent + 1);

        }
        os << std::endl << std::setw(indent + 1) << "}";

    }

    return os;
}

std::ostream &operator<<(std::ostream &os, Properties const &prop)
{
    prop.print(os, 0);
    return os;
}

};//namespace simpla