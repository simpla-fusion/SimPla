//
// Created by salmon on 16-11-2.
//

#ifndef SIMPLA_PRINTABLE_H
#define SIMPLA_PRINTABLE_H

#include <ostream>

namespace simpla { namespace data
{

struct Printable
{
    Serializable() {}

    virtual ~Serializable() {}

    virtual std::string const &name() const =0;

    virtual std::ostream &print(std::ostream &os, int indent) const =0;


};

std::ostream &operator<<(std::ostream &os, Printable const &obj)
{
    os << obj.name() << " = {";
    obj.print(os, 1);
    os << "}" << std::endl;
    return os;
}

}}
#endif //SIMPLA_PRINTABLE_H
