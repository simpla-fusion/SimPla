/*
 * field_test.h
 *
 *  created on: 2014-6-30
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

template<typename TM, unsigned int IFORM, typename TV>
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
		LOGGER.set_stdout_visable_level(LOG_DEBUG);

		for (unsigned int i = 0; i < NDIMS; ++i)
		{
			if (dims[i] <= 1 || xmax[i] <= xmin[i])
			{
				xmax[i] = xmin[i];
				dims[i] = 1;
			}
		}
		mesh.set_dimensions(dims);
		mesh.set_extents(xmin, xmax);
		mesh.update();

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

	coordinates_type xmin = { 10, 0, 0 };

	coordinates_type xmax = { 12, 1, 1 };

	nTuple<NDIMS, index_type> dims = { 5, 6, 10 };
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

	std::memset(&v,0,sizeof(value_type));

	field_type f(mesh,v );

	EXPECT_EQ(v,f.default_value());

	EXPECT_EQ(0,f.size());

	f.clear();

	if(field_type::is_dense_storage)
	{
		EXPECT_EQ(mesh.get_local_memory_size(IForm),f.size());
	}
	else
	{
		EXPECT_EQ(0,f.size());
	}

	// Check default value

	for(auto s:mesh.Select(IForm))
	{
		EXPECT_EQ(v,f[s]);
	}

}
}

TYPED_TEST_P(TestField,assign){
{
	typedef typename TestFixture::field_type field_type;
	typedef typename TestFixture::mesh_type mesh_type;
	typedef typename TestFixture::value_type value_type;

	typedef typename mesh_type::scalar_type scalar_type;
	static constexpr unsigned int NDIMS = TestFixture::NDIMS;
	static constexpr unsigned int IForm = TestFixture::IForm;

	mesh_type const & mesh = TestFixture::mesh;

	auto f=(mesh.template make_field<field_type>( ));

	f.clear();

	for(auto s:mesh.Select(IForm))
	{
		value_type ss;
		ss=s;
		f[s]=ss;
	}

	EXPECT_EQ(mesh.get_local_memory_size(IForm),f.size());

	for(auto s:mesh.Select(IForm))
	{
		value_type ss;
		ss=s;

		EXPECT_EQ(ss,f[s]);
	}

}
}

TYPED_TEST_P(TestField,traversal){
{
	typedef typename TestFixture::field_type field_type;
	typedef typename TestFixture::mesh_type mesh_type;
	typedef typename TestFixture::value_type value_type;

	typedef typename mesh_type::scalar_type scalar_type;
	static constexpr unsigned int NDIMS = TestFixture::NDIMS;
	static constexpr unsigned int IForm = TestFixture::IForm;

	mesh_type const & mesh = TestFixture::mesh;

	auto f=(mesh.template make_field<field_type>( ));

	f.clear();

	size_t count =0;

	for(auto s:mesh.Select(IForm))
	{
		value_type ss;
		ss=count;
		f[s]=ss;
		++count;
	}

	EXPECT_EQ(mesh.get_local_memory_size(IForm),f.size());

	count=0;

	for(auto s:mesh.Select(field_type::IForm) )
	{
		value_type ss;
		ss=count;
		EXPECT_EQ(ss,f[s]);
		++count;
	}

}
}

REGISTER_TYPED_TEST_CASE_P(TestField, create, assign, traversal);

#endif /* FIELD_TEST_H_ */
