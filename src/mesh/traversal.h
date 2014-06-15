/*
 * traversal.h
 *
 *  Created on: 2014年3月20日
 *      Author: salmon
 */

#ifndef TRAVERSAL_H_
#define TRAVERSAL_H_

#include <thread>
#include <vector>
#include "../utilities/sp_type_traits.h"
#include "../fetl/cache.h"

namespace simpla
{

template<int IFORM, typename IT, typename TF, typename ...Args>
void Traversal(IT it, IT ie, TF &&fun, Args && ...args)
{
	while (true)
	{
		fun(*it, args...);

		++it;
		if (it == ie)
			break;
	}

}

template<typename IT, typename TF, typename ... Args>
void ParallelFor(Range<IT> range, std::function<void(typename IT::value_type, Args const & ...)> fun,
		Args const & ...args)
{

	const unsigned int num_threads = std::thread::hardware_concurrency();

	std::vector<std::thread> threads;

	for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
	{
		threads.emplace_back(

		std::thread(

		[num_threads,thread_id](Args const & ... args2)
		{	for(auto s: range.split(num_threads,thread_id))
			{
				fun(s,std::forward<Args const &>(args2)...);
			}
		}, std::forward<Args const &>(args)...)

		);
	}

	for (auto & t : threads)
	{
		t.join();
	}

}

template<int IFORM, typename TC, typename TF, typename ... Args>
void ParallelCachedTraversal(TC const &tree, TF &&fun, Args && ...args)
{
	const unsigned int num_threads = std::thread::hardware_concurrency();

	std::vector<std::thread> threads;

	for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
	{
		auto Range = tree.GetRegin(IFORM, num_threads, thread_id);

		threads.emplace_back(

		std::thread(

		[Range](TF fun2,typename Cache<Args>::type && ... args2 )
		{
			for (auto s:Range)
			{
				RefreshCache(s,args2...);

				fun2(s,args2...);

				FlushCache(s,args2...);
			}

		}, fun, typename Cache<Args >::type(args)...

		)

		);
	}

	for (auto & t : threads)
	{
		t.join();
	}
}

}  // namespace simpla

#endif /* TRAVERSAL_H_ */
