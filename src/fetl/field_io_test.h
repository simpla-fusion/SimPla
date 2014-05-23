/*
 * field_io_test.h
 *
 *  Created on: 2014年5月12日
 *      Author: salmon
 */

#ifndef FIELD_IO_TEST_H_
#define FIELD_IO_TEST_H_
#include <gtest/gtest.h>
#include "save_field.h"
namespace simpla
{
template<typename TParam>
class TestFieldIO: public testing::Test
{

protected:
	virtual void SetUp()
	{
		TParam::SetUpMesh(&mesh);
		TParam::SetDefaultValue(&default_value);
		GLOBAL_DATA_STREAM.OpenFile("FetlTest");
		GLOBAL_DATA_STREAM.OpenGroup("/");
	}
public:

	typedef typename TParam::mesh_type mesh_type;
	typedef typename TParam::value_type value_type;
	typedef typename mesh_type::iterator iterator;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef Field<mesh_type, VERTEX, value_type> TZeroForm;
	typedef Field<mesh_type, EDGE, value_type> TOneForm;
	typedef Field<mesh_type, FACE, value_type> TTwoForm;
	typedef Field<mesh_type, VOLUME, value_type> TThreeForm;

	mesh_type mesh;

	static constexpr double PI = 3.141592653589793;

	static constexpr nTuple<3, Real> k =
	{	2.0 * PI, 2.0 * PI, 4.0 * PI}; // @NOTE must   k = n TWOPI, period condition

	value_type default_value;

};
TYPED_TEST_CASE_P(TestFieldIO);

TYPED_TEST_P(TestFieldIO, write){
{
	auto const & mesh= TestFixture::mesh;

	typename TestFixture::TTwoForm f2(mesh);
	typename TestFixture::TTwoForm f2b(mesh);
	typename TestFixture::TThreeForm f3(mesh);

	f3.Clear();
	f2.Clear();
	f2b.Clear();

	LOGGER<<SAVE(f3);
	LOGGER<<SAVE(f2);
	LOGGER<<SAVE(f2b);
}
}

}
 // namespace simpla

#endif /* FIELD_IO_TEST_H_ */
