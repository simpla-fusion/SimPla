/*
 * mesh_test.h
 *
 *  Created on: 2014年3月25日
 *      Author: salmon
 */

#ifndef MESH_TEST_H_
#define MESH_TEST_H_

#include <gtest/gtest.h>

#include "../utilities/pretty_stream.h"
#include "../utilities/log.h"
#include "../physics/constants.h"
#include "../io/data_stream.h"
#include "../parallel/message_comm.h"

using namespace simpla;

template<typename TMesh>
class TestMesh: public testing::Test
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

		mesh.SetExtents(xmin,xmax,dims);

		if( !GLOBAL_DATA_STREAM.IsOpened())
		{
			GLOBAL_DATA_STREAM.OpenFile("MeshTest");
			GLOBAL_DATA_STREAM.OpenGroup("/");
		}
	}
public:
	typedef TMesh mesh_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::range_type range_type;
	typedef typename mesh_type::iterator iterator;
	typedef typename TMesh::coordinates_type coordinates_type;

	unsigned int NDIMS=TMesh::NDIMS;

	mesh_type mesh;

	std::vector<unsigned int> iform_list =
	{	VERTEX, EDGE, FACE, VOLUME};

	coordinates_type xmin,xmax;

	nTuple<TMesh::NDIMS, index_type> dims;

};

TYPED_TEST_CASE_P(TestMesh);

TYPED_TEST_P(TestMesh,scatter ){
{

//	Field<mesh_type, VERTEX, Real> n(mesh);
//	Field<mesh_type, EDGE, Real> J(mesh);
//
//	n.Clear();
//	J.Clear();
//
//	nTuple<3, Real> x =
//	{	-0.01, -0.01, -0.01};
//	nTuple<3, Real> v =
//	{	1, 2, 3};
//
//	for (auto const & v : n)
//	{
//		std::cout << " " << v;
//	}
//	std::cout << std::endl;
//	for (auto const & v : J)
//	{
//		std::cout << " " << v;
//	}
//	std::cout << std::endl;
//	mesh.Scatter(x, 1.0, &n);
//	mesh.Scatter(x, v, &J);
//	for (auto const & v : n)
//	{
//		std::cout << " " << v;
//	}
//	std::cout << std::endl;
//	for (auto const & v : J)
//	{
//		std::cout << " " << v;
//	}
//
//	std::cout << std::endl;

}}

TYPED_TEST_P(TestMesh,gather){
{

}
}

REGISTER_TYPED_TEST_CASE_P(TestMesh, scatter, gather);


#endif /* MESH_TEST_H_ */
