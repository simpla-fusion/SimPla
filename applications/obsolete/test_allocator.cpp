/*
 * test_allocator.cpp
 *
 *  Created on: 2013年12月9日
 *      Author: salmon
 */

#include "../src/utilities/memory_pool.h"
#include "../src/utilities/log.h"

#include <vector>
#include <list>
#include <memory>
#include <functional>
#include "../src/fetl/fetl.h"
#include "../src/mesh/co_rect_mesh.h"
using namespace simpla;
DEFINE_FIELDS(CoRectMesh<>)
int main(int argc, char **argv)
{

	Mesh mesh;
	mesh.dt_ = 1.0;
	mesh.xmin_[0] = 0;
	mesh.xmin_[1] = 0;
	mesh.xmin_[2] = 0;
	mesh.xmax_[0] = 1.0;
	mesh.xmax_[1] = 1.0;
	mesh.xmax_[2] = 1.0;
	mesh.dims_[0] = 20;
	mesh.dims_[1] = 30;
	mesh.dims_[2] = 40;

	mesh.Update();

	MEMPOOL.SetPoolSizeInGB(20);
	CHECK(MEMPOOL.GetMemorySizeInGB()) << "GB";
	{
		Form<1> f1(mesh);
		CHECK(MEMPOOL.GetMemorySizeInGB()) << "GB";
		f1 = 0.0;
		CHECK(MEMPOOL.GetMemorySizeInGB()) << "GB";

		for (int i = 0; i < 10; ++i)
		{
			Form<2> f2(mesh);
			CHECK(MEMPOOL.GetMemorySizeInGB()) << "GB";
		}

		for (int i = 0; i < 10; ++i)
		{
			Form<2> * f2 = new Form<2>(mesh);

			*f2 = 0.0;
			delete f2;
		}
	}
	CHECK(MEMPOOL.GetMemorySizeInGB()) << "GB";

}

