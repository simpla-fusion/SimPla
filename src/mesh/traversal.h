/*
 * traversal.h
 *
 *  Created on: 2014年3月20日
 *      Author: salmon
 */

#ifndef TRAVERSAL_H_
#define TRAVERSAL_H_

#include <thread>
namespace simpla
{

template<int IFORM, typename TC, typename TF, typename ...Args>
void Traversal(TC const & tree, TF &&fun, Args && ...args)
{
	auto it = tree.begin(IFORM), ie = tree.end(IFORM);

	while (true)
	{
		fun(*it, args...);

		++it;
		if (it == ie)
			break;
	}

}

template<int IFORM, typename TC, typename TF, typename ... Args>
void ParallelTraversal(TC const & tree, TF &&fun, Args && ...args)
{
	Traversal<IFORM>(tree, fun, args...);
//	const unsigned int num_threads = std::thread::hardware_concurrency();
//
//	std::vector<std::thread> threads;
//
////	for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
////	{
////		auto ib = tree.begin(IFORM, num_threads, thread_id);
////		auto ie = tree.end(IFORM, num_threads, thread_id);
////
////		threads.emplace_back(
////
////		std::thread(
////
////		[&]( )
////		{
////			for (auto it =ib; it != ie; ++it)
////			{
////				fun (*it,args ...);
////			}
////
////		}
////
////		));
////	}
//
//	for (auto & t : threads)
//	{
//		t.join();
//	}
}

template<int IFORM, typename TC, typename TF, typename ... Args>
void ParallelCachedTraversal(TC const &tree, TF &&fun, Args && ...args)
{
	const unsigned int num_threads = std::thread::hardware_concurrency();

	std::vector<std::thread> threads;

	for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
	{
		auto ib = tree.begin(IFORM, num_threads, thread_id);
		auto ie = tree.end(IFORM, num_threads, thread_id);

		threads.emplace_back(

		std::thread(

		[ib,ie](TF fun2,typename Cache<Args>::type && ... args2 )
		{
			for (auto it =ib; it != ie; ++it)
			{
				RefreshCache(*it,args2...);

				fun2(*it,args2...);

				FlushCache(*it,args2...);
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
