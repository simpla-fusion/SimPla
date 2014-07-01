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

template<typename TMesh>
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
	typedef TMesh mesh_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::compact_index_type compact_index_type;
	typedef typename mesh_type::range_type range_type;
	typedef typename mesh_type::iterator iterator;
	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::scalar_type scalar_type;
	static constexpr unsigned int NDIMS = TMesh::NDIMS;

	mesh_type mesh;

	coordinates_type xmin =
	{	10,0,0};

	coordinates_type xmax =
	{	12,1,1};

	nTuple<NDIMS, index_type> dims =
	{	20,30,1};

};

TYPED_TEST_CASE_P(TestIterpolator);

TYPED_TEST_P(TestIterpolator,vertex){
{

	typedef typename TestFixture::mesh_type mesh_type;
	typedef typename TestFixture::compact_index_type compact_index_type;
	typedef typename TestFixture::coordinates_type coordinates_type;
	typedef typename TestFixture::scalar_type scalar_type;

	auto const & mesh=TestFixture::mesh;

	static constexpr unsigned int NDIMS=TestFixture::NDIMS;

	auto extents= mesh.GetExtents();

	coordinates_type x = mesh.InvMapTo(std::get<0>(extents)+0.1234567*(std::get<1>(extents)-std::get<0>(extents)));

	Real w=2;

	scalar_type a=3.1415926;

	auto f= mesh.template make_field<Field<mesh_type,VERTEX,SparseContainer<compact_index_type,scalar_type>>> ();

	mesh.Scatter(Int2Type<VERTEX>(),&f,std::make_tuple(x,a),w);

	auto b= mesh.Gather(Int2Type<VERTEX>(),f,x);

	EXPECT_LE(abs(a*w-b),EPSILON);

}
}

TYPED_TEST_P(TestIterpolator,edge){
{

	typedef typename TestFixture::mesh_type mesh_type;
	typedef typename TestFixture::compact_index_type compact_index_type;
	typedef typename TestFixture::coordinates_type coordinates_type;
	typedef typename TestFixture::scalar_type scalar_type;

	auto const & mesh=TestFixture::mesh;

	static constexpr unsigned int NDIMS=TestFixture::NDIMS;

	auto extents= mesh.GetExtents();

	coordinates_type x = mesh.InvMapTo(std::get<0>(extents)+0.1234567*(std::get<1>(extents)-std::get<0>(extents)));

	Real w=2;

	nTuple<NDIMS,scalar_type> a=
	{	3.1415926 , -3.1415926,3.0*3.1415926};

	auto f= mesh.template make_field<Field<mesh_type,EDGE,SparseContainer<compact_index_type,scalar_type>>>( );

	mesh.Scatter(Int2Type<EDGE>(),&f,std::make_tuple(x,a),w);

	auto b= mesh.Gather(Int2Type<EDGE>(),f,x);

	EXPECT_LE(abs(a[0]*w-b[0]),EPSILON);
	EXPECT_LE(abs(a[1]*w-b[1]),EPSILON);
	EXPECT_LE(abs(a[2]*w-b[2]),EPSILON);

}
}

TYPED_TEST_P(TestIterpolator,face){
{

	typedef typename TestFixture::mesh_type mesh_type;
	typedef typename TestFixture::compact_index_type compact_index_type;
	typedef typename TestFixture::coordinates_type coordinates_type;
	typedef typename TestFixture::scalar_type scalar_type;

	auto const & mesh=TestFixture::mesh;

	static constexpr unsigned int NDIMS=TestFixture::NDIMS;

	auto extents= mesh.GetExtents();

	coordinates_type x = mesh.InvMapTo(std::get<0>(extents)+0.1234567*(std::get<1>(extents)-std::get<0>(extents)));

	Real w=2;

	nTuple<NDIMS,scalar_type> a=
	{	3.1415926 , -3.1415926,3.0*3.1415926};

	auto f= mesh.template make_field<Field<mesh_type,FACE,SparseContainer<compact_index_type,scalar_type>>> ( );

	mesh.Scatter(Int2Type<FACE>(),&f,std::make_tuple(x,a),w);

	auto b=mesh.Gather(Int2Type<FACE>(),f,x);

	EXPECT_LE(abs(a[0]*w-b[0]),EPSILON);
	EXPECT_LE(abs(a[1]*w-b[1]),EPSILON);
	EXPECT_LE(abs(a[2]*w-b[2]),EPSILON);

}
}

TYPED_TEST_P(TestIterpolator,volume){
{

	typedef typename TestFixture::mesh_type mesh_type;
	typedef typename TestFixture::compact_index_type compact_index_type;
	typedef typename TestFixture::coordinates_type coordinates_type;
	typedef typename TestFixture::scalar_type scalar_type;

	auto const & mesh=TestFixture::mesh;

	static constexpr unsigned int NDIMS=TestFixture::NDIMS;

	auto extents= mesh.GetExtents();

	coordinates_type x = mesh.InvMapTo(std::get<0>(extents)+0.1234567*(std::get<1>(extents)-std::get<0>(extents)));

	Real w=2;

	scalar_type a=3.1415926;

	auto f= mesh.template make_field<Field<mesh_type,VOLUME,SparseContainer<compact_index_type,scalar_type>>> ( );

	mesh.Scatter(Int2Type<VOLUME>(),&f,std::make_tuple(x,a),w);

	auto b= mesh.Gather(Int2Type<VOLUME>(),f,x);

	EXPECT_LE(abs(a*w-mesh.Gather(Int2Type<VOLUME>(),f,x)),EPSILON);

}
}

REGISTER_TYPED_TEST_CASE_P(TestIterpolator, vertex, edge, face, volume);

#endif /* ITERPOLATOR_TEST_H_ */
