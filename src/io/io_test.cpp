/*
 * io_test.cpp
 *
 *  Created on: 2012-3-16
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include "include/simpla_defs.h"
#include "fetl/fetl.h"
#include "io/io.h"

using namespace simpla;

class TestIO: public testing::Test
{
protected:
	virtual void SetUp()
	{
		IVec3 dims =
		{ 20, 30, 40 };

		Vec3 xmin =
		{ 0, 0, 0 };

		Vec3 xmax =
		{ 1, 1, 1 };

		grid.Initialize(1.0, xmin, xmax, dims);

	}
public:
	Grid grid;
};

TYPED_F(TestIO,create_write_read)
{
	typename TestFixture::Grid const & grid = TestFixture::grid;

	typename TestFixture::FieldType f(grid);

	size_t size =grid.get_num_of_vertex()
	* grid.get_num_of_comp (TestFixture::FieldType::IForm);

	typename TestFixture::FieldType::ValueType a = 1.0;

	for (size_t s = 0; s < size; ++s)
	{
		f[s] = a*static_cast<typename TestFixture::ValueType>(s);
	}

	for (size_t s = 0; s<size; ++s)
	{
		ASSERT_EQ(a*static_cast<Real>(s),f[s])<<"idx=" << s;
	}

}

