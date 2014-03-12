/*
 * media_tag_test.cpp
 *
 *  Created on: 2014年3月12日
 *      Author: salmon
 */
#include <gtest/gtest.h>
#include <random>
#include <string>
#include "../io/data_stream.h"
#include "../fetl/save_field.h"

#include "material.h"
#include "../mesh/rect_mesh.h"
#include "../mesh/octree_forest.h"
#include "../fetl/fetl.h"

using namespace simpla;
template<typename TF>
class TestMaterial: public testing::Test
{
protected:
	virtual void SetUp()
	{
		Logger::Verbose(10);

		nTuple<NDIMS, Real> xmin = { 0, 0, 0 };
		nTuple<NDIMS, Real> xmax = { 1, 1, 1 };
		mesh.SetExtent(xmin, xmax);

		nTuple<NDIMS, size_t> dims = { 200, 200, 0 };
		mesh.SetDimensions(dims);
		mesh.Update();

		dims = mesh.GetDimensions();

		materials = std::shared_ptr<material_type>(new material_type(mesh));

		for (int i = 0; i < NDIMS; ++i)
		{
			dh[i] = (dims[i] > 1) ? (xmax[i] - xmin[i]) / dims[i] : 0;
		}

	}
public:
	typedef TF field_type;
	typedef typename TF::mesh_type mesh_type;

	static constexpr unsigned int NDIMS = mesh_type::NDIMS;
	static constexpr unsigned int IForm = TF::IForm;

	typedef Material<mesh_type> material_type;

	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type mesh;
	std::shared_ptr<material_type> materials;

	nTuple<NDIMS, Real> dh;

};

typedef RectMesh<OcForest> Mesh;

typedef testing::Types<

Field<Mesh, VERTEX, Real>,

Field<Mesh, EDGE, Real>,

Field<Mesh, FACE, Real>,

Field<Mesh, VOLUME, Real>

> MeshTypes;

TYPED_TEST_CASE(TestMaterial, MeshTypes);

TYPED_TEST(TestMaterial,create ){
{

//	std::mt19937 gen;
//	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);
	typedef typename TestFixture::mesh_type mesh_type;
	typedef typename TestFixture::index_type index_type;
	typedef typename TestFixture::coordinates_type coordinates_type;
	typename TestFixture::mesh_type const & mesh=TestFixture::mesh;
	typename TestFixture::material_type material(TestFixture::mesh);

	auto extent=mesh.GetExtent();

	std::vector<coordinates_type> v;

	v.emplace_back(coordinates_type(
					{	0.2*extent.second[0], 0.2*extent.second[1], 0.2*extent.first[2]}));

	v.emplace_back(coordinates_type(
					{	0.8*extent.second[0], 0.8*extent.second[1], 0.2*extent.first[2]}));

	material.Add("Plasma",v);

	material.Update();

	typename TestFixture::field_type f(mesh);

	f.Clear();

	material.template SelectCell<TestFixture::IForm>(
			[&](index_type const & s ,coordinates_type const & x)
			{	f[s]=1;},"Plasma" );

	coordinates_type v0,v1,v2,v3;
	for (int i = 0; i <TestFixture:: NDIMS; ++i)
	{
		v0[i] = v[0][i]+TestFixture::dh[i];
		v1[i] = v[1][i]-TestFixture::dh[i];

		v2[i] = v[0][i]-TestFixture::dh[i]*2;
		v3[i] = v[1][i]+TestFixture::dh[i]*2;
	}
	mesh.template Traversal<TestFixture::IForm>(
			[&](index_type s)
			{
				auto x=mesh.GetCoordinates(s);

				if( ((((v0[0]-x[0])*(x[0]-v1[0]))>=0)&&
								(((v0[1]-x[1])*(x[1]-v1[1]))>=0)&&
								(((v0[2]-x[2])*(x[2]-v1[2]))>=0))
				)
				{
					EXPECT_TRUE(f[s]==1)<< (mesh.GetCoordinates(s));
				}

				if( !(((v2[0]-x[0])*(x[0]-v3[0]))>=0)&&
						(((v2[1]-x[1])*(x[1]-v3[1]))>=0)&&
						(((v2[2]-x[2])*(x[2]-v3[2]))>=0))
				{
					EXPECT_FALSE(f[s]==1)<< (mesh.GetCoordinates(s));
				}
			}

	);

	v.emplace_back(coordinates_type(
					{	0.3*extent.second[0], 0.6*extent.second[1], 0.2*extent.first[2]}));

	material.Remove("Plasma",v );

	f.Clear();

	material.template SelectCell<TestFixture::IForm>(
			[&](index_type const & s ,coordinates_type const & x)
			{	f[s]=1;},"Plasma" );
	GLOBAL_DATA_STREAM.OpenFile("MaterialTest");
	GLOBAL_DATA_STREAM.OpenGroup("/");
	std::cout<<Dump(f,"f" ,false)<<std::endl;
}
}
