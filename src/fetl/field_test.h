/*
 * field_test.h
 *
 *  Created on: 2014年6月30日
 *      Author: salmon
 */

#ifndef FIELD_TEST_H_
#define FIELD_TEST_H_
#include <gtest/gtest.h>

#include "../utilities/log.h"
#include "../utilities/pretty_stream.h"
#include "../utilities/ntuple.h"

#include "../physics/constants.h"

#include "field.h"

using namespace simpla;

template<typename TM, int IFORM, typename TV>
struct ParamType
{
	typedef TM mesh_type;
	static constexpr unsigned int IForm = IFORM;
	typedef TV value_type;
};

template<typename TF>
class TestField: public testing::Test
{
protected:
	void SetUp()
	{
		LOG_STREAM.SetStdOutVisableLevel(LOG_DEBUG);

		for (int i = 0; i < NDIMS; ++i)
		{
			if (dims[i] <= 1 || xmax[i] <= xmin[i])
			{
				xmax[i] = xmin[i];
				dims[i] = 1;
			}
		}

		mesh.SetExtents(xmin, xmax, dims);

	}
public:
	typedef TF field_type;
	typedef typename field_type::mesh_type mesh_type;
	typedef typename field_type::value_type value_type;
	static constexpr unsigned int IForm = field_type::IForm;
	static constexpr unsigned int NDIMS = mesh_type::NDIMS;

	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::compact_index_type compact_index_type;
	typedef typename mesh_type::range_type range_type;
	typedef typename mesh_type::iterator iterator;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::scalar_type scalar_type;

	mesh_type mesh;

	coordinates_type xmin =
	{	10, 0, 0};

	coordinates_type xmax =
	{	12, 1, 1};

	nTuple<NDIMS, index_type> dims =
	{	20, 30, 1};
};

TYPED_TEST_CASE_P(TestField);

TYPED_TEST_P(TestField,create){
{
	typedef typename TestFixture::field_type field_type;
	typedef typename TestFixture::mesh_type mesh_type;
	typedef typename TestFixture::value_type value_type;

	typedef typename mesh_type::scalar_type scalar_type;
	static constexpr unsigned int NDIMS = TestFixture::NDIMS;
	static constexpr unsigned int IForm = TestFixture::IForm;

	mesh_type const & mesh = TestFixture::mesh;

	value_type v;

	v=3.14145926;

	auto f=(mesh.template clone_field<field_type>(v));

	EXPECT_EQ(v,f.default_value());

	EXPECT_EQ(0,f.size());

	f.initialize();

	if(field_type::is_dense_storage)
	{
		EXPECT_EQ(mesh.GetLocalMemorySize(IForm),f.size());
	}
	else
	{
		EXPECT_EQ(0,f.size());
	}

	for(auto s:mesh.Select(IForm))
	{
		EXPECT_EQ(v,f[s]);
	}

}
}

TYPED_TEST_P(TestField,assign){
{
//	typedef typename TestFixture::mesh_type mesh_type;
//	typedef typename TestFixture::value_type value_type;
//
//	typedef typename mesh_type::scalar_type scalar_type;
//	static constexpr unsigned int NDIMS = TestFixture::NDIMS;
//	static constexpr unsigned int IForm = TestFixture::IForm;
//
//	mesh_type const & mesh = TestFixture::mesh;
//
//	auto f0=mesh.template make_form<IForm,value_type>();
//
//	for(auto s:mesh.Select(IForm))
//	{
//
//	}
}
}

REGISTER_TYPED_TEST_CASE_P(TestField, create, assign);

#endif /* FIELD_TEST_H_ */
