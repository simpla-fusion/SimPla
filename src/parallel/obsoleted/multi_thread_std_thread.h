/**
 * @file multi_thread_std_thread.h
 *
 * @date    2014-8-28  AM8:11:49
 * @author salmon
 */

#ifndef MULTI_THREAD_STD_THREAD_H_
#define MULTI_THREAD_STD_THREAD_H_

#include <vector>
#include <iostream>

#include <thread>
#include <future>


namespace simpla
{
//template<typename TI>
//RangeHolder<TI> make_divisible_range(TI const & b, TI const &e)
//{
//	size_t grainsize = b - e;
//
//	int max_thread = std::thread::hardware_concurrency();
//
//	if (grainsize > max_thread * 2)
//	{
//		grainsize = grainsize / max_thread + 1;
//	}
//
//	return std::move(RangeHolder<TI>(b, e, grainsize));
//}
//
//template<typename TRange, typename Func>
//void parallel_for(TRange && entity_id_range, Func && fun)
//{
//
//	tbb::parallel_for(std::forward<Arrgs>(entity_id_range), std::forward<Func>(fun));
////	if (!is_divisible(std::forward<TRange>(entity_id_range)))
////	{
////		fun(std::forward<TRange>(entity_id_range));
////	}
////	else
////	{
////
////		auto t_r = split(std::forward<TRange>(entity_id_range));
////
////		auto f1 = std::async(std::launch::async, [&]()
////		{	parallel_for( std::get<0>(t_r), std::forward<Func>(fun));});
////
////		parallel_for(std::get<1>(t_r), fun);
////
////		f1.get();
////	}
//}
//
////
/////**
//// *  @ingroup MULTICORE
//// *
//// * \brief Parallel for each
//// * @param r
//// * @param fun
//// */
//template<typename TRange, typename Func>
//void parallel_for_each(TRange && entity_id_range, Func && fun)
//{
//
//	parallel_for(std::forward<TRange>(entity_id_range),
//
//	[&](TRange const & r1)
//	{
//		for( auto const & v:r1 )
//		{	fun(v);}
//
//	});
//
//}

/**
 *  @ingroup MULTICORE
 *
 * \brief Parallel do
 * @param fun
 */
/**
 *
 * @param entity_id_range
 * @param fun void(TRange::value_type,TRes*)
 * @param res
 * @param red_fun
 */
template<typename TRange, typename TRes, typename Func, typename Reduction>
void parallel_reduce(TRange && range, TRes const & identity, TRes *res,
		Func &&fun, Reduction &&reduction)
{

	if (!is_divisible(std::forward<TRange>(range)))
	{
		fun(std::forward<TRange>(range), res);
	}
	else
	{

		TRes tmp(identity);

		auto t_r = split(std::forward<TRange>(range));

		auto f1 =
				std::async(std::launch::async,
						[&]()
						{
							parallel_reduce( std::get<0>(t_r),identity,&tmp,
									std::forward<Func>(fun), std::forward<Reduction>(reduction ));
						});

		parallel_reduce(std::get<1>(t_r), identity, res,
				std::forward<Func>(fun), std::forward<Reduction>(reduction));

		f1.get();

		reduction(tmp, res);

	}

}
template<typename TRange, typename TRes, typename Func>
void parallel_reduce(TRange && range, TRes const & identity, TRes *res,
		Func const &fun)
{
	parallel_reduce(std::forward<TRange>(range), res, fun, [](TRes &l,TRes *r)
	{	*r+=l;});
}
}
// namespace simpla

#endif /* MULTI_THREAD_STD_THREAD_H_ */
