/**
 * @file calculate.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_CALCULATE_H
#define SIMPLA_CALCULATE_H


#include "../calculus.h"

namespace simpla
{

template<typename ...> struct Calculate;

namespace calculate
{
namespace tags
{
struct finite_difference;
struct finite_volume;
struct DG;

}//namespace tags
}//namespace calculate
}//namespace simpla
#endif //SIMPLA_CALCULATE_H
