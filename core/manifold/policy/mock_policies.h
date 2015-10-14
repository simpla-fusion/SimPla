/**
 * @file calculate_mock.h
 * @author salmon
 * @date 2015-10-13.
 */

#ifndef SIMPLA_CALCULATE_MOCK_H
#define SIMPLA_CALCULATE_MOCK_H

#include "../../gtl/type_traits.h"
#include "../../field/field_traits.h"

namespace simpla
{

namespace tags
{
struct is_mock;
}
namespace policy
{


template<typename ...> struct calculate;
template<typename ...> struct interpolator;

template<typename TM>
struct calculate<TM, tags::is_mock>
{
	virtual TM const &geometry() const= 0;

	template<typename TF, typename ...Args> traits::value_type_t <traits::value_type_t<TF>>
	eval(TM const &, TF const &, Args &&...args) const
	{
		traits::value_type_t<traits::value_type_t<TF> > res;
		return std::move(res);
	}
};

template<typename TM>
struct interpolator<TM, tags::is_mock>
{
	virtual TM const &geometry() const= 0;

	template<int I, typename TF, typename ...Args> traits::value_type_t <traits::value_type_t<TF>>
	sample(TM const &, TF const &, Args &&...args) const
	{
		traits::value_type_t<traits::value_type_t<TF>> res;
		return std::move(res);
	}

	template<typename TF, typename ...Args>
	traits::value_type_t <traits::field_value_t<TF>>
	gather(TM const &, TF const &, Args &&...args) const
	{
		traits::value_type_t<traits::field_value_t<TF>> res;
		return std::move(res);
	}

	template<typename ...Args>
	void scatter(Args &&...args) const
	{
	}

};
}//namespace policy
}//namespace simpla
#endif //SIMPLA_CALCULATE_MOCK_H
