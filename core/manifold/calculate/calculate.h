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

namespace traits
{

template<typename ... T>
struct type_id<Calculate<T...> >
{
	static std::string name()
	{
		return "Calculate< \"" + type_id<T...>::name() + "\" >";
	}
};

DEFINE_TYPE_ID_NAME(calculate::tags::finite_difference)

DEFINE_TYPE_ID_NAME(calculate::tags::finite_volume)

DEFINE_TYPE_ID_NAME(calculate::tags::DG)

}//namespace traits
}//namespace simpla
#endif //SIMPLA_CALCULATE_H
