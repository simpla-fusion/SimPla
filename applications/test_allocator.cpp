/*
 * test_allocator.cpp
 *
 *  Created on: 2013年12月9日
 *      Author: salmon
 */

#include "../src/utilities/allocator_mempool.h"
#include "../src/utilities/log.h"

namespace stl
_GLIBCXX_VISIBILITY(default)
{
	_GLIBCXX_BEGIN_NAMESPACE_VERSION
	template<typename T, typename _ForwardIterator, typename _Size>
	void __uninitialized_default_n_a(_ForwardIterator __first, _Size __n,
			simpla::MemPoolAllocator<T>& __alloc)
	{
		CHECK("It works!");
	}
	_GLIBCXX_END_NAMESPACE_VERSION
}
// namespace stl
#include <vector>
#include <list>
#include <memory>
#include <functional>

using namespace simpla;

int main(int argc, char **argv)
{
	MEMPOOL.SetPoolSizeInGB(20);

	std::vector<double, MemPoolAllocator<double> > v1;
	v1.reserve(100);
	CHECK(v1.size());
	CHECK(MEMPOOL.GetMemorySizeInGB())<<"GB";

	std::vector<double, MemPoolAllocator<double> > v2;
	v2.reserve(10000000);
	CHECK(MEMPOOL.GetMemorySizeInGB())<<"GB";

	for(int i=0;i<10;++i)
	{
		auto *v3=new std::vector<double, MemPoolAllocator<double> >();
		v3->reserve(100000000L);
		CHECK(MEMPOOL.GetMemorySizeInGB())<<"GB";
		CHECK(v3->size());
		delete v3;
		CHECK(MEMPOOL.GetMemorySizeInGB())<<"GB";
	}

}

