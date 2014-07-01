/*
 * iterpolator_test.h
 *
 *  Created on: 2014年6月29日
 *      Author: salmon
 */

#ifndef ITERPOLATOR_TEST_H_
#define ITERPOLATOR_TEST_H_

#include <gtest/gtest.h>

#include "../utilities/pretty_stream.h"
#include "../utilities/log.h"
#include "../physics/constants.h"
#include "../io/data_stream.h"
#include "../parallel/message_comm.h"
#include "../fetl/field.h"
#include "../utilities/container_sparse.h"

using namespace simpla;

template<typename TP>
class TestIterpolator: public testing::Test
{
protected:
	void SetUp()
	{
		LOG_STREAM.SetStdOutVisableLevel(10);

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
	typedef typename TP::mesh_type mesh_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::compact_index_type compact_index_type;
	typedef typename mesh_type::range_type range_type;
	typedef typename mesh_type::iterator iterator;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::scalar_type scalar_type;
	static constexpr unsigned int NDIMS = mesh_type::NDIMS;
	static constexpr unsigned int IForm = TP::IForm;
	mesh_type mesh;

	coordinates_type xmin =
	{	0,0,0};

	coordinates_type xmax =
	{	1,1,1};

	nTuple<NDIMS, index_type> dims =
	{	50,30,20};

};

TYPED_TEST_CASE_P(TestIterpolator);

TYPED_TEST_P(TestIterpolator,scatter){
{

	auto const & mesh=TestFixture::mesh;
	static constexpr unsigned int IForm = TestFixture::IForm;
	static constexpr unsigned int NDIMS = TestFixture::NDIMS;
	typedef typename TestFixture::mesh_type mesh_type;
	typedef typename TestFixture::compact_index_type compact_index_type;
	typedef typename TestFixture::coordinates_type coordinates_type;
	typedef typename TestFixture::scalar_type scalar_type;

	auto f= mesh.template make_field<Field<mesh_type,IForm,SparseContainer<compact_index_type,scalar_type>>> ();

	typename decltype(f)::field_value_type a;

	a=1.0;

	Real w=2;

	auto extents= mesh.GetExtents();

	coordinates_type x = mesh.InvMapTo(std::get<0>(extents)+0.1234567*(std::get<1>(extents)-std::get<0>(extents)));

	mesh.Scatter(Int2Type<IForm>(),&f,std::make_tuple(x,a),w);

	scalar_type b=0;

	SparseContainer<compact_index_type,scalar_type>& g=f;

	for(auto const & v:g)
	{
		b+=v.second;
	}

	if(IForm==VERTEX || IForm==VOLUME)
	{
		EXPECT_LE(abs( w-b),EPSILON*10);
	}
	else
	{
		EXPECT_LE(abs( w*3-b),EPSILON*10);
	}

}
}

TYPED_TEST_P(TestIterpolator,gather){
{

	auto const & mesh=TestFixture::mesh;
	static constexpr unsigned int IForm = TestFixture::IForm;
	static constexpr unsigned int NDIMS = TestFixture::NDIMS;
	typedef typename TestFixture::mesh_type mesh_type;
	typedef typename TestFixture::compact_index_type compact_index_type;
	typedef typename TestFixture::coordinates_type coordinates_type;
	typedef typename TestFixture::scalar_type scalar_type;

	auto f= mesh.template make_field<IForm,scalar_type > ();

	f.clear();

	auto extents= mesh.GetExtents();

	nTuple<NDIMS,Real> K=
	{	TWOPI,PI, PI};

	for(auto s:mesh.Select(IForm))
	{
		f[s]=std::cos(InnerProductNTuple(K,mesh.GetCoordinates(s)-std::get<0>(extents)));
	}
	coordinates_type x = mesh.InvMapTo(std::get<0>(extents)+0.34567*(std::get<1>(extents)-std::get<0>(extents)));

	Real expect=std::cos(InnerProductNTuple(K,x));

	Real error = abs(std::pow(InnerProductNTuple(K , mesh.GetDx()) , 2.0));

	auto actual= mesh.Gather(Int2Type<IForm>(), f ,x);

	CHECK(actual);
	CHECK(expect);

	EXPECT_LE(abs( (actual -expect)/expect),error)<<actual <<" "<<expect;

//	nTuple<NDIMS,scalar_type> a=
//	{	3.1415926 , -3.1415926,3.0*3.1415926};
//	typedef typename TestFixture::mesh_type mesh_type;
//	typedef typename TestFixture::compact_index_type compact_index_type;
//	typedef typename TestFixture::coordinates_type coordinates_type;
//	typedef typename TestFixture::scalar_type scalar_type;
//
//	auto const & mesh=TestFixture::mesh;
//
//	static constexpr unsigned int NDIMS=TestFixture::NDIMS;
//
//	auto extents= mesh.GetExtents();
//
//	coordinates_type x = mesh.InvMapTo(std::get<0>(extents)+0.1234567*(std::get<1>(extents)-std::get<0>(extents)));
//
//	Real w=2;
//
//	scalar_type a=1.0;
//
//	auto f= mesh.template make_field<Field<mesh_type,VERTEX,SparseContainer<compact_index_type,scalar_type>>> ();
//
//	mesh.Scatter(Int2Type<VERTEX>(),&f,std::make_tuple(x,a),w);
//
//	scalar_type b=0;
//	SparseContainer<compact_index_type,scalar_type>& g=f;
//
//	for(auto const & v:g)
//	{
//		b+=v.second;
//	}
//
//	EXPECT_LE(abs(a*w-b),EPSILON);
//
//	for(auto & v:g)
//	{
//		v.second=a;
//	}
//
//	EXPECT_LE(abs(a - mesh.Gather(Int2Type<VERTEX>(),f,x)),EPSILON);

}
}

REGISTER_TYPED_TEST_CASE_P(TestIterpolator, scatter, gather);

#endif /* ITERPOLATOR_TEST_H_ */
