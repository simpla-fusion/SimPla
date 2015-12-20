/*
 * field_io_test.h
 *
 *  created on: 2014-5-12
 *      Author: salmon
 */

#ifndef FIELD_IO_TEST_H_
#define FIELD_IO_TEST_H_

#include <gtest/gtest.h>

#include "../../gtl/utilities/log.h"
#include "../../io/IO.h"
#include "manifold_traits.h"

namespace simpla
{
template<typename TParam>
class TestFieldIO : public testing::Test
{

protected:
	virtual void SetUp()
	{
		TParam::SetUpMesh(&mesh);
		TParam::SetDefaultValue(&default_value);
		GLOBAL_DATA_STREAM.open("FetlTest.h5:/");
	}

public:

	typedef typename TParam::mesh_type mesh_type;
	typedef typename TParam::value_type value_type;
	typedef typename mesh_type::iterator iterator;
	typedef typename mesh_type::coordinate_tuple coordinate_tuple;

	mesh_type mesh;

	static constexpr double PI = 3.141592653589793;

	static constexpr nTuple<3, Real> k =
			{2.0 * PI, 2.0 * PI, 4.0 * PI}; // @NOTE must   k = n TWOPI, period condition

	value_type default_value;

};

TYPED_TEST_CASE_P(TestFieldIO);

TYPED_TEST_P(TestFieldIO, write)
{
	{
		auto const &mesh = TestFixture::mesh;
		typedef typename TestFixture::value_type value_type;

		auto f2 = mesh.make_field<FACE, value_type>();
		auto f2b = mesh.make_field<FACE, value_type>();
		auto f3 = mesh.make_field<VOLUME, value_type>();

		f3.clear();
		f2.clear();
		f2b.clear();

		LOGGER << SAVE(f3);
		LOGGER << SAVE(f2);
		LOGGER << SAVE(f2b);

		LOGGER << simpla::save("f3", f3, true);
		LOGGER << simpla::save("f3", f3, true);
		LOGGER << simpla::save("f3", f3, true);
	}
}

}
// namespace simpla

#endif /* FIELD_IO_TEST_H_ */
