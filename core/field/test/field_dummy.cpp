/*
 * field_dummy.cpp
 *
 *  Created on: 2015-1-27
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include <tuple>

#include "../../utilities/utilities.h"
#include "../../io/IO.h"
#include "field_comm.h"
using namespace simpla;

template<typename TV, typename TM>
class FieldTest: public testing::TestWithParam<TM>
{

protected:
	void SetUp()
	{
		LOGGER.set_stdout_visable_level(LOG_VERBOSE);

		std::tie(xmin, xmax, dims, K_real) = GetParam();

		K_imag = 0;
	}

	void TearDown()
	{
		std::shared_ptr<mesh_type>(nullptr).swap(mesh);
	}
public:

	typedef TM mesh_type;
	typedef TV value_type;

	typedef typename mesh_type::scalar_type scalar_type;
	typedef typename mesh_type::coordinate_tuple coordinate_tuple;

	mesh_type mesh;

	virtual ~FieldTest()
	{
	}

};

