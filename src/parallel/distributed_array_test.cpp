/*
 * distributed_array_test.cpp
 *
 *  Created on: 2014年5月30日
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include "distributed_array.h"
#include "../fetl/ntuple.h"
#include "../utilities/pretty_stream.h"
#include <stddef.h>
#include "update_ghosts.h"

using namespace simpla;

class TestDistArray: public testing::TestWithParam<nTuple<3, size_t> >
{

protected:
	virtual void SetUp()
	{
		global_count = GetParam();
		global_start = 10000;

	}
public:
	nTuple<3, size_t> global_start;
	nTuple<3, size_t> global_count;
	static constexpr unsigned int NDIMS = 3;
	DistributedArray<NDIMS> darray;
};

//TEST_P(TestDistArray, Init)
//{
//	darray.Init(3, 1, 2, global_start, global_count);
//
//	CHECK(global_start);
//	CHECK(global_count);
//	CHECK(darray.local_.outer_start);
//	CHECK(darray.local_.outer_count);
//	CHECK(darray.local_.inner_start);
//	CHECK(darray.local_.inner_count);
//
//}

TEST_P(TestDistArray, UpdateGhost)
{
	GLOBAL_COMM.Init();

	darray.global_start_=global_start;
	darray.global_count_=global_count;

	darray.Decompose(GLOBAL_COMM.GetSize(), GLOBAL_COMM.GetRank(), 2);

	std::vector<double> data(darray.memory_size());

	std::fill(data.begin(), data.end(),GLOBAL_COMM.GetRank());
	size_t count =0;
	for(auto & v:data)
	{
		v=count+(GLOBAL_COMM.GetRank()+1)*1000;
		++count;
	}

	UpdateGhosts(&data[0],darray);

	MPI_Barrier( GLOBAL_COMM.GetComm());

	if(GLOBAL_COMM.GetRank()==0)
	{
		count =0;
		for(auto const & v:data)
		{
			if((count%darray.local_.outer_count[1])==0)
			{
				std::cout<<std::endl<<"["<< GLOBAL_COMM.GetRank()<<"/"<<GLOBAL_COMM.GetSize()<<"]";
			}

			std::cout<<v<<" ";

			++count;
		}
		std::cout<<std::endl;
	}
	MPI_Barrier( GLOBAL_COMM.GetComm());
}

INSTANTIATE_TEST_CASE_P(Parallel, TestDistArray, testing::Values(nTuple<3, size_t>(
{ 10, 20, 1 })));
