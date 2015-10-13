/**
 * @file calculate_mock.h
 * @author salmon
 * @date 2015-10-13.
 */

#ifndef SIMPLA_CALCULATE_MOCK_H
#define SIMPLA_CALCULATE_MOCK_H
namespace simpla
{
namespace policy
{

struct MockCalculate
{
	template<int I, typename TF, typename ...Args> traits::value_type_t <traits::value_type_t<TF>>
	proxy_sample(TF const &, Args &&...args) const
	{
		traits::value_type_t<traits::value_type_t<TF>> res;
		return std::move(res);
	}

	template<typename TM, typename TF, typename ...Args> traits::value_type_t <traits::value_type_t<TF>>
	proxy_calculate(TM const &, TF const &, Args &&...args) const
	{
		traits::value_type_t<traits::value_type_t<TF> > res;
		return std::move(res);
	}

	template<typename TF, typename ...Args> traits::value_type_t <traits::field_value_t<TF>>
	proxy_gather(TF const &, Args &&...args) const
	{
		traits::value_type_t<traits::field_value_t<TF>> res;
		return std::move(res);
	}

	template<typename ...Args>
	void proxy_scatter(Args &&...args) const
	{
	}
//	template<typename ...Args>
//	auto sample(Args &&...args) const
//	DECL_RET_TYPE((interpolate_policy::template sample<iform>(*this, std::forward<Args>(args)            ...)))
//
//
//	template<typename ...Args>
//	auto gather(Args &&...args) const
//	DECL_RET_TYPE((interpolate_policy::gather(*this, std::forward<Args>(args)...)))
//
//
//	template<typename ...Args>
//	auto calculate(Args &&...args) const
//	DECL_RET_TYPE((calculate_policy::eval(*this, std::forward<Args>(args)...)))
//
//	template<typename ...Args>
//	void scatter(Args &&...args) const
//	{
//		interpolate_policy::scatter(*this, std::forward<Args>(args)...);
//	}

};

}//namespace policy
}//namespace simpla
#endif //SIMPLA_CALCULATE_MOCK_H
