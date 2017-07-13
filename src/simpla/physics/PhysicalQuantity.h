//
// Created by salmon on 16-10-5.
//

#ifndef SIMPLA_PHYSICALQUANTITY_H
#define SIMPLA_PHYSICALQUANTITY_H

#include "simpla/_config.h"
#include <string>

namespace simpla { namespace physic
{
/**
 *
 */
class PhysicalQuantity
{
    std::string name;
    size_type m_id_;
    Real m_si_[6];
    size_type m_num_of_dims_;

};
}}
#endif //SIMPLA_PHYSICALQUANTITY_H
