/*
 * field_dummy.cpp
 *
 *  Created on: 2015年1月27日
 *      Author: salmon
 */

#include <gtest/gtest.h>
#include <tuple>

#include "../../utilities/utilities.h"
#include "../../io/io.h"
#include "../field.h"
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
	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type mesh;

	virtual ~FieldTest()
	{
	}

};

