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
#include "../mesh/mesh_rectangle.h"
#include "../mesh/octree_forest.h"
#include "../mesh/geometry_euclidean.h"
#include "../fetl/fetl.h"

using namespace simpla;
template<typename TParam>
class TestMaterial: public testing::Test
{
protected:
	virtual void SetUp()
	{
		Logger::Verbose(10);

		TParam::SetUpMesh(&mesh);
		auto dims = mesh.GetDimensions();

		materials = std::shared_ptr<material_type>(new material_type(mesh));
		auto extent = mesh.GetExtent();
		for (int i = 0; i < NDIMS; ++i)
		{
			dh[i] = (dims[i] > 1) ? (extent.second[i] - extent.first[i]) / dims[i] : 0;
		}

		GLOBAL_DATA_STREAM.OpenFile("MaterialTest");
		GLOBAL_DATA_STREAM.OpenGroup("/");
	}
public:

	typedef typename TParam::mesh_type mesh_type;
	typedef typename TParam::value_type value_type;
	static constexpr unsigned int IForm = TParam::IForm;
	typedef Field<mesh_type, IForm, value_type> field_type;

	static constexpr unsigned int NDIMS = mesh_type::NDIMS;

	typedef Material<mesh_type> material_type;
	typedef typename mesh_type::index_type index_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type mesh;
	std::shared_ptr<material_type> materials;

	nTuple<NDIMS, Real> dh;

};

TYPED_TEST_CASE_P(TestMaterial);

TYPED_TEST_P(TestMaterial,create ){
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

	material.Set("Plasma",v);
	material.Update();

	typename TestFixture::field_type f(mesh);

	f.Clear();

	for(auto s:material.Select(mesh.begin( TestFixture::IForm ),mesh.end( TestFixture::IForm ),"Plasma" ))
	{
		f[s]=1;
	}
	LOGGER<<DUMP(f);

	coordinates_type v0,v1,v2,v3;
	for (int i = 0; i <TestFixture:: NDIMS; ++i)
	{
		v0[i] = v[0][i]+TestFixture::dh[i];
		v1[i] = v[1][i]-TestFixture::dh[i];

		v2[i] = v[0][i]-TestFixture::dh[i]*2;
		v3[i] = v[1][i]+TestFixture::dh[i]*2;
	}
	for (auto s : mesh.GetRange(TestFixture::IForm))
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

	v.emplace_back(coordinates_type(
					{	0.3*extent.second[0], 0.6*extent.second[1], 0.2*extent.first[2]}));

	material.Remove("Plasma",v );
	material.Update();

	f.Clear();

	for(auto s: material.Select ( mesh.begin(TestFixture::IForm ),mesh.end(TestFixture::IForm ) ,"Plasma" ))
	{
		f[s]=1;
	}

	LOGGER<<DUMP(f );

	for(auto s: material.Select ( mesh.begin(TestFixture::IForm ),mesh.end(TestFixture::IForm ) ,"Plasma" ,"NONE"))
	{
		f[s]=10;
	}

//	for(auto s:material.Select( mesh.begin(TestFixture::IForm ),mesh.end(TestFixture::IForm ) , "Vacuum","Plasma" ))
//	{
//		f[s]=-10;
//	}
	LOGGER<<DUMP(f );
}
}

REGISTER_TYPED_TEST_CASE_P(TestMaterial, create);

template<typename TM, typename TV, int IFORM> struct TestFETLParam;

template<typename TV, int IFORM>
struct TestFETLParam<RectMesh<OcForest, EuclideanGeometry>, TV, IFORM>
{
	typedef RectMesh<OcForest, EuclideanGeometry> mesh_type;
	typedef TV value_type;
	static constexpr int IForm = IFORM;

	static void SetUpMesh(mesh_type * mesh)
	{

		nTuple<3, Real> xmin = { 0, 0, 0 };
		nTuple<3, Real> xmax = { 1, 1, 1 };
		mesh->SetExtent(xmin, xmax);

		nTuple<3, size_t> dims = { 200, 200, 0 };
		mesh->SetDimensions(dims, true);
		mesh->Update();
	}

};

typedef RectMesh<OcForest, EuclideanGeometry> mesh_type;

typedef testing::Types<

TestFETLParam<mesh_type, Real, VERTEX>,

TestFETLParam<mesh_type, Real, EDGE>  ,

TestFETLParam<mesh_type, Real, FACE>,

TestFETLParam<mesh_type, Real, VOLUME>

> ParamList;

INSTANTIATE_TYPED_TEST_CASE_P(MATERIAL, TestMaterial, ParamList);
