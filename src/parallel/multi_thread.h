/*
 * multi_thread.h
 *
 *  created on: 2014-5-12
 *      Author: salmon
 */

#ifndef MULTI_THREAD_H_
#define MULTI_THREAD_H_
#include <vector>
#include <iostream>
#ifdef _OPENMP
#	include <omp.h>
#else
#	include <thread>
#   include <future>
#endif

#include "../utilities/sp_range.h"
namespace simpla
{

int get_num_concurrency(unsigned int num_threads_hint = 0)
{
#ifdef _OPENMP

	if (num_threads_hint == 0)
	{
		num_threads_hint = omp_get_num_procs();
	}

	omp_set_num_threads(num_threads_hint);
//	num_threads_hint = omp_get_num_threads();

#else

	if (num_threads_hint == 0)
	{
		num_threads_hint = std::thread::hardware_concurrency();
	}

#endif

	return num_threads_hint;
}

template<typename TRange>
void parallel_for(TRange & range, std::function<TRange &> const & fun, size_t grain_size = 0)
{

	if (grain_size == 0)
	{
		grain_size = size(range) / get_num_concurrency() + 1;
	}

	if (size(range) <= grain_size)
	{
		fun(range);
	}
	else
	{

#ifdef _OPENMP
#pragma omp parallel
		{
			parallel_for(split(range,omp_get_num_threads(),omp_get_thread_num()),fun,grain_size);
		}
#else

		auto t_r = split(range);

		auto f1 = std::async(std::launch::async, parallel_for<TRange>, std::get<0>(t_r), fun, grain_size);

		parallel_do(std::get<1>(t_r), fun, grain_size);

		f1.get();
#endif
	}
}

//
///**
// *  \ingroup MULTICORE
// *
// * \brief Parallel for each
// * @param r
// * @param fun
// */
template<typename TRange, typename TF>
void parallel_for_each(TRange & range, TF const & fun, size_t grain_size = 0)
{

	parallel_for(range,

	[&fun](TRange & r1)
	{
		for( auto & v:r1 )
		{	fun(v);}

	},

	grain_size);

}

/**
 *  \ingroup MULTICORE
 *
 * \brief Parallel do
 * @param fun
 */
/**
 *
 * @param range
 * @param fun void(TRange::value_type,TRes*)
 * @param res
 * @param red_fun
 */
template<typename TRange, typename TRes, typename TFun, typename Reduction>
void parallel_reduce(TRange const & range, TRes *res, TFun const &fun, Reduction const &red_fun,
        std::size_t grain_size = 0)
{
	if (grain_size == 0)
	{
		grain_size = size(range) / get_num_concurrency() + 1;

	}

	if (size(range) <= grain_size)
	{
		fun(range, res);
	}
	else
	{

		TRes tmp(*res);

		auto t_r = split(range);

		auto f1 = std::async(std::launch::async, [&]()
		{
			parallel_reduce( std::get<0>(t_r),&tmp, fun, red_fun, grain_size);

		});

		parallel_reduce(std::get<1>(t_r), res, fun, red_fun, grain_size);

		f1.get();

		red_fun(tmp, res);

	}

}

//template<typename TRange, typename TFun, typename TRes>
//void parallel_reduce(TRange const &range, TFun const& fun, TRes *res)
//{
//
//	parallel_reduce(range, fun, res,
//
//	[](TRes const& t_res,TRes *res2)
//	{
//		*res2+= t_res;
//	}
//
//	);
//
//}
}
// namespace simpla

#endif /* MULTI_THREAD_H_ */
