/*
 * integration.h
 *
 *  Created on: 2013-12-2
 *      Author: salmon
 */

#ifndef INTEGRATION_H_
#define INTEGRATION_H_

#include <map>

namespace simpla
{

template<typename TX, typename TY>
inline std::map<TX, decltype( std::declval<TX>()*std::declval<TY>())> //
Integrate(std::map<TX, TY> const & xy)
{
	typedef decltype( std::declval<TX>()*std::declval<TY>()) res_value_type;
	typedef std::map<TX, res_value_type> res_type;
	typedef typename std::map<TX, TY>::iterator iterator;
	typedef TX x_type;
	typedef TY y_type;

	res_value_type f = 0;

	res_type res;

	res.emplace(std::make_pair(xy.begin()->first, f));

	for (iterator it1 = xy.begin(), it2 = it1++; it2 != xy.end(); ++it1, ++it2)
	{
		f += 0.5 * (it1->second + it2->second) * (it2->first - it1->first);
		res.emplace(std::make_pair(it2->first, f));
	}

	return std::move(res);
}

}
// namespace simpla

#endif /* INTEGRATION_H_ */
