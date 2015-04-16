/**
 * @file demo_misc2.cpp
 *
 * @date 2015年4月16日
 * @author salmon
 */

#include "../../core/gtl/iterator/sp_ntuple_range.h"
#include "../../core/gtl/ntuple.h"
#include "../../core/utilities/utilities.h"
#include "../../core/mesh/mesh_ids.h"

#include <iostream>

using namespace simpla;

int main(int argc, char **argv)
{
	nTuple<size_t, 3> b = { 0, 0, 0 };
	nTuple<size_t, 3> e = { 2, 3, 4 };

	auto range = make_ntuple_range(b, e);

	size_t count = 0;

	range.is_slow_first(true);

	for (auto v : range)
	{
		std::cout << v << std::endl;

		++count;
	}
	std::cout << "Count = " << count << std::endl;

	range.is_slow_first(false);

	count = 0;

	for (auto v : range)
	{
		std::cout << v << std::endl;

		++count;
	}
	std::cout << "Count = " << count << std::endl;

	MeshIDs_<3>::range_type<VERTEX> r2(b, e);

	CHECK(*r2.begin());

//	count = 0;
//
//	for (auto v : r2)
//	{
//		std::cout << v << std::endl;
//
//		++count;
//	}
//	std::cout << "Count = " << count << std::endl;

}
