//
// Created by salmon on 16-11-2.
//

#ifndef SIMPLA_PRINTABLE_H
#define SIMPLA_PRINTABLE_H

#include <ostream>

namespace simpla { namespace toolbox
{

struct Printable
{
    Printable() {}

    virtual ~Printable() {}

    virtual std::string const &name() const =0;

    virtual std::ostream &print(std::ostream &os, int indent) const =0;


};

inline std::ostream &operator<<(std::ostream &os, Printable const &obj)
{
    obj.print(os, 0);
    return os;
}

}}
#endif //SIMPLA_PRINTABLE_H
