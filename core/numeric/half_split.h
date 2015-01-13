/*
 * half_split.h
 *
 *  Created on: 2014年12月10日
 *      Author: salmon
 */

#ifndef CORE_NUMERIC_HALF_SPLIT_H_
#define CORE_NUMERIC_HALF_SPLIT_H_

namespace simpla
{
template<typename T>
std::tuple<T, T> half_split(std::tuple<T, T> & range)
{
	std::tuple<T, T> res = range;
	range = tmp / 2;
	return tmp - range;
}
template<typename T, size_t N>
nTuple<T, N> half_split(nTuple<T, N> & range)
{
	nTuple<T, N> res;
	res = range;

	auto n = max_at(res);

	res[n] = half_split(range[n]);

	return std::move(res);
}
//template<typename T>
//std::vector<std::tuple<T, T>> split(T const start, T const & count, size_t num)
//{
//	nTuple<T, N> res;
//	res = range;
//
//	auto n = max_at(res);
//
//	res[n] = half_split(range[n]);
//
//	return std::move(res);
//}
//template<typename T, size_t N>
//std::vector<nTuple<T, N>> split(nTuple<T, N> const & range, size_t num)
//{
//	nTuple<T, N> res;
//	res = range;
//
//	auto n = max_at(res);
//
//	res[n] = half_split(range[n]);
//
//	return std::move(res);
//}
}  // namespace simpla

#endif /* CORE_NUMERIC_HALF_SPLIT_H_ */
