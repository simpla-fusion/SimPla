/*
 * model_test.cpp
 *
 *  Created on: 2014年3月12日
 *      Author: salmon
 */
#include <gtest/gtest.h>
#include <random>
#include <string>
#include "../io/data_stream.h"
#include "../fetl/save_field.h"

#include "model.h"
#include "../mesh/mesh_rectangle.h"
#include "../mesh/octree_forest.h"
#include "../mesh/geometry_euclidean.h"
#include "../fetl/fetl.h"

using namespace simpla;
template<typename TParam>
class TestModel: public testing::Test
{
protected:
	virtual void SetUp()
	{
		TParam::SetUpMesh(&mesh);
		auto dims = mesh.GetDimensions();

		model = std::shared_ptr<model_type>(new model_type(mesh));
		auto extent = mesh.GetExtents();
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

	typedef Model<mesh_type> model_type;
	typedef typename mesh_type::iterator iterator;
	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type mesh;
	std::shared_ptr<model_type> model;

	nTuple<NDIMS, Real> dh;

};

TYPED_TEST_CASE_P(TestModel);

TYPED_TEST_P(TestModel,create ){
{

//	std::mt19937 gen;
//	std::uniform_real_distribution<Real> uniform_dist(0, 1.0);

	typedef typename TestFixture::mesh_type mesh_type;
	typedef typename TestFixture::iterator iterator;
	typedef typename TestFixture::coordinates_type coordinates_type;
	typename TestFixture::mesh_type const & mesh=TestFixture::mesh;
	typename TestFixture::model_type model(TestFixture::mesh);

	auto extent=mesh.GetExtents();

	std::vector<coordinates_type> v;

	v.emplace_back(coordinates_type(
					{	0.2*extent.second[0], 0.2*extent.second[1], 0.2*extent.first[2]}));

	v.emplace_back(coordinates_type(
					{	0.8*extent.second[0], 0.8*extent.second[1], 0.2*extent.first[2]}));

	model.Set("Plasma",v);
	model.Update();

	typename TestFixture::field_type f(mesh);

	f.Clear();

	for(auto s:model.Select(mesh.Select( TestFixture::IForm ),"Plasma" ))
	{
		f[s]=1;
	}
	LOGGER<<SAVE(f);

	coordinates_type v0,v1,v2,v3;
	for (int i = 0; i <TestFixture:: NDIMS; ++i)
	{
		v0[i] = v[0][i]+TestFixture::dh[i];
		v1[i] = v[1][i]-TestFixture::dh[i];

		v2[i] = v[0][i]-TestFixture::dh[i]*2;
		v3[i] = v[1][i]+TestFixture::dh[i]*2;
	}
	for (auto s : mesh.Select(TestFixture::IForm))
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

	model.Remove("Plasma",v );
	model.Update();

	f.Clear();

	for(auto s: model.Select ( mesh.Select( TestFixture::IForm ) ,"Plasma" ))
	{
		f[s]=1;
	}

	LOGGER<<SAVE(f );

	for(auto s: model.Select ( mesh.Select( TestFixture::IForm ) ,"Plasma" ,"NONE"))
	{
		f[s]=10;
	}

//	for(auto s:model.Select( mesh.Select( TestFixture::IForm ) , "Vacuum","Plasma" ))
//	{
//		f[s]=-10;
//	}
	LOGGER<<SAVE(f );
}
}

REGISTER_TYPED_TEST_CASE_P(TestModel, create);

template<typename TM, typename TV, int IFORM> struct TestModelParam;

template<typename TV, int IFORM>
struct TestModelParam<Mesh<EuclideanGeometry<OcForest>>, TV, IFORM>
{
	typedef Mesh<EuclideanGeometry<OcForest>> mesh_type;
	typedef TV value_type;
	static constexpr int IForm = IFORM;

	static void SetUpMesh(mesh_type * mesh)
	{

		nTuple<3, Real> xmin = { 0, 0, 0 };
		nTuple<3, Real> xmax = { 1, 1, 1 };

		nTuple<3, size_t> dims = { 200, 200, 0 };

		mesh->SetExtents(xmin, xmax, dims);
	}

};

typedef Mesh<EuclideanGeometry<OcForest>> mesh_type;

typedef testing::Types<

TestModelParam<mesh_type, Real, VERTEX>,

TestModelParam<mesh_type, Real, EDGE>,

TestModelParam<mesh_type, Real, FACE>,

TestModelParam<mesh_type, Real, VOLUME>

> ParamList;

INSTANTIATE_TYPED_TEST_CASE_P(MATERIAL, TestModel, ParamList);
