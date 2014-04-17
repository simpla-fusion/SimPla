/*
 * test_stdcxx_allcoator.cpp
 *
 *  Created on: 2013年12月22日
 *      Author: salmon
 */

#include <ext/mt_allocator.h>
#include <iostream>
#include <list>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include "../src/utilities/utilities.h"

template<typename T> using FixedSmallSizeAlloc=__gnu_cxx::__mt_alloc<T>;

struct Foo
{
	std::string str;

	Foo(std::string const & s)
			: str(s)
	{
		std::cout << "I'm constructed! " + str + "\n" << std::endl;
	}
	Foo(Foo const & s)
			: str(s.str)
	{
		std::cout << "I'm copyed! " + str + "\n" << std::endl;
	}

	Foo(Foo && s)
			: str(s.str)
	{
		std::cout << "I'm moved! " + str + "\n" << std::endl;
	}
	~Foo()
	{
		std::cout << "I'm destructed! " + str + "\n" << std::endl;
	}
};

int main(int argc, char **argv)
{
	typedef std::list<Foo, FixedSmallSizeAlloc<Foo> > list_type;

//	unsigned int num_threads = std::thread::hardware_concurrency();
//
//	std::vector<list_type> l(num_threads);
//
//	std::vector<std::thread> threads;
//
//	for (unsigned int thread_id = 0; thread_id < num_threads; ++thread_id)
//	{
//		threads.emplace_back(std::thread([thread_id,&l]()
//		{
//			Foo s(simpla::ToString(std::this_thread::get_id()));
//			l[thread_id].push_back(s);
//		}));
//	}
//
//	std::cout << " I'm the master!";
//
//	for (auto & t : threads)
//	{
//		t.join();
//	}

	list_type l2;
	Foo s("WaWa!");
	for (int i = 0; i < 10; ++i)
	{
		l2.push_back(s);
	}
	std::cout << " =========================" << std::endl;
	for (int i = 0; i < 10; ++i)
	{
		l2.erase(l2.begin());
	}
	std::cout << " =========================" << std::endl;

}

