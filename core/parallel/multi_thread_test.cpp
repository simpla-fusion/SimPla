/**
 * \file multi_thread_test.cpp
 *
 * \date    2014年8月27日  上午7:25:40 
 * \author salmon
 */
#include <iostream>
#include <unordered_map>
//#include <tbb/tbb.h>
//#include <tbb/concurrent_unordered_map.h>
struct my_hash
{
	int operator()(int i) const
	{
		return i % 2;
	}
};
int main(int argc, char **argv)
{
	std::unordered_multimap<int, int, my_hash> a;

	for (int i = 0; i < 2048; ++i)
	{
		a.emplace(i, i);
		std::cout << a.bucket_count() << " " << a.max_load_factor() << std::endl;
	}

//	for (int i = 0; i < a.bucket_count(); ++i)
//	{
//		for (auto ib = a.begin(i), ie = a.end(i); ib != ie; ++ib)
//		{
//
//		}
//	}

//	for (auto &v : a)
//	{
//		v *= 2;
//
//	}
//	for (int i = 0; i < a.bucket_count(); ++i)
//	{
//		std::cout << "[" << i << "] " << a.bucket_size(i) << std::endl;
//
////		for (auto ib = a.begin(i), ie = a.end(i); ib != ie; ++ib)
////		{
////			std::cout << *ib << " ";
////		}
////		std::cout << std::endl;
//	}

//	typedef tbb::concurrent_unordered_map<int, int> container;
//	typedef typename container::iterator iterator;
//	typedef typename container::range_type range_type;
//	container a;
//
//	for (int i = 0; i < 100; ++i)
//	{
//		a[i] = 2 * i;
//	}
//
//	tbb::parallel_for(a.range(),
//
//	[](range_type const &r )
//	{
//		for(auto ib=r.begin(),ie=r.end();ib!=ie;++ib)
//		{
//			std::cout<< ib->first<<","<<ib->second<<std::endl;
//		}
//	}
//
//	);

//	std::vector<int> d(100);
//
//	int total = 0;
//
//	auto range = make_divisible_range(d.begin(), d.end());
//
//	typedef decltype(range) range_type;
//
//	parallel_for(range, [](range_type &r )
//	{	for(auto & v:r)
//		{	v=2;}});
//
//	parallel_reduce(range, 0, &total,
//
//	[](range_type const &r, int *res )
//	{	for(auto const & v:r)
//		{	*res+=v;}},
//
//	[](int l,int *r)
//	{	*r+=l;}
//
//	);
//
//	std::cout << total << std::endl;
//
////	parallel_do([](int num_threads,int thread_num)
////	{
////		std::cout<< thread_num <<" in " << num_threads<<std::endl;
////	}, 4);
//
////	std::vector<int> data(10);
////	for (auto & v : data)
////	{
////		v = 1;
////	}
////	int total = 0;
////	std::function<void(int const &, int*)> foo = [](int const &a ,int*b)
////	{	*b+=a;};
////
////	parallel_reduce(std::make_pair(data.begin(), data.end()), foo, &total);
////	std::cout << " total = " << total << std::endl;
//	auto f1 = std::async(std::launch::async, [&]()
//	{
//		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
//		std::cout<<"This is 1"<<std::endl;
//
//	});
//
//	auto f2 = std::async(std::launch::async, [&]()
//	{
//		std::this_thread::sleep_for(std::chrono::milliseconds(2000));
//		std::cout<<"This is 2"<<std::endl;
//
//	});
//
//	std::this_thread::sleep_for(std::chrono::milliseconds(1600));
//			std::cout<<"This is 0"<<std::endl;
//
//	f1.get();
//	f2.get();

}

