/*
 * inverse_function.h
 *
 *  Created on: 2013-12-2
 *      Author: salmon
 */

#ifndef INVERSE_FUNCTION_H_
#define INVERSE_FUNCTION_H_
#include <map>
namespace simpla
{

template<typename TX, typename TY>
std::map<TY, TX> Inverse(std::map<TX, TY> const & xy)
{
	std::map<TY, TX> res;
	for (auto const &p : xy)
	{
		res.emplace(std::make_pair(p.second, p.first));
	}

	return std::move(res);

}

}  // namespace simpla

#endif /* INVERSE_FUNCTION_H_ */
