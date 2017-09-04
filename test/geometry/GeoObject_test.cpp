/*
 * model_test.cpp
 *
 *  created on: 2014-3-12
 *      Author: salmon
 */

#include <gtest/gtest.h>

#include "simpla/data/SPObject.h"
#include "simpla/geometry/GeoObject.h"
using namespace simpla;
struct DummyGeoObject : public simpla::geometry::GeoObject {
    SP_OBJECT_HEAD(DummyGeoObject, SPObject)
    SP_OBJECT_PROPERTY(Real, Mass);
    SP_OBJECT_PROPERTY(Real, Charge);
};
DummyGeoObject::DummyGeoObject() = default;
DummyGeoObject::~DummyGeoObject() = default;

std::shared_ptr<simpla::data::DataNode> DummyGeoObject::Serialize() const { return base_type::Serialize(); };
void DummyGeoObject::Deserialize(std::shared_ptr<const data::DataNode> const& cfg) { base_type::Deserialize(cfg); };
SP_OBJECT_REGISTER(DummyGeoObject)
TEST(SPObject, Dummy) {
    auto objA = DummyGeoObject::New();
    objA->SetMass(1.0);
    objA->SetCharge(-1.0);
    std::cout << *objA->Serialize() << std::endl;

    auto objB = std::dynamic_pointer_cast<DummyGeoObject>(SPObject::Create(objA->Serialize()));
    EXPECT_TRUE(objB != nullptr);
    EXPECT_DOUBLE_EQ(objA->GetMass(), objB->GetMass());
    EXPECT_DOUBLE_EQ(objA->GetCharge(), objB->GetCharge());
}
//#include "../../geometry/geometry.h"
//
//#include <gtest/gtest.h>
//
//#include <random>
//#include <string>
//
//#include "../../diff_geometry/diff_scheme/fdm.h"
//#include "../../diff_geometry/fetl.h"
//#include "../../diff_geometry/geometry/cartesian.h"
//#include "../../diff_geometry/interpolator/interpolator.h"
//#include "../../diff_geometry/topology/structured.h"
//#include "../utilities/type_traits.h"
//
// using namespace simpla;
//
// template<typename TInt>
// class TestModel: public testing::Test
//{
// protected:
//	virtual void InitializeConditionPatch()
//	{
//
//		xmin = nTuple<Real, ndims>( { 0.0, 0.0, 0.0 });
//
//		xmax = nTuple<Real, ndims>( { 1.0, 2.0, 3.0 });
//
//		dims = nTuple<size_t, ndims>( { 50, 60, 10 });
//
//		geometry.dimensions(dims);
//
//		geometry.extents(xmin, xmax);
//
//		geometry.update();
//
//		auto extent = geometry.extents();
//
//		for (int i = 0; i < ndims; ++i)
//		{
//			dh[i] = (dims[i] > 1) ?
//					(extent.m_node_[i] - extent.first[i]) / dims[i] : 0;
//		}
//
//		points.emplace_back(
//				coordinate_tuple(
//						{ 0.2 * xmax[0], 0.2 * xmax[1], 0.2 * xmin[2] }));
//
//		points.emplace_back(
//				coordinate_tuple(
//						{ 0.8 * xmax[0], 0.8 * xmax[1], 0.8 * xmax[2] }));
//
//		LOGGER.set_MESSAGE_visable_level(12);
////		GLOBAL_DATA_STREAM.cd("MaterialTest.h5:/");
//	}
// public:
//
//	typedef Manifold<CartesianCoordinate<RectMesh>, FiniteDiffMethod,
//			InterpolatorLinear> manifold_type;
//	typedef Real value_type;
//	typedef typename manifold_type::coordinate_tuple coordinate_tuple;
//	typedef Model<manifold_type> model_type;
//
//	static constexpr size_t GetIFORM = TInt::value;
//	static constexpr size_t ndims = manifold_type::ndims;
//
//	nTuple<Real, ndims> xmin/* = { 0.0, 0.0, 0.0, }*/;
//
//	nTuple<Real, ndims> xmax/* = { 1.0, 2.0, 3.0 }*/;
//
//	nTuple<size_t, ndims> dims/* = { 50, 60, 10 }*/;
//
//	model_type geometry;
//
//	nTuple<Real, ndims> dh;
//
//	std::vector<coordinate_tuple> points;
//
//};
// TYPED_TEST_CASE_P(TestModel);
//
// TYPED_TEST_P(TestModel,SelectByNGP){
//{
//	auto & geometry= TestFixture::geometry;
//
//	typedef typename TestFixture::manifold_type manifold_type;
//
//	typename TestFixture::coordinate_tuple min,max,x;
//
//	std::tie(min,max)=geometry.extents();
//
//	x=min*0.7+max*0.3;
//
//	typename manifold_type::index_type dest;
//
//	std::tie(dest,std::ignore)=geometry.coordinates_global_to_local(x);
//
//	auto GetRange=geometry.SelectByNGP( make_domain<TestFixture::GetIFORM>(geometry), x);
//
//	size_t size =0;
//
//	for(auto s :GetRange)
//	{
//		EXPECT_EQ( manifold_type::get_cell_index(s),manifold_type::get_cell_index(dest));
//		++size;
//	}
//
//	LOGGER<<size;
//
//	EXPECT_EQ(size,manifold_type::get_num_of_comp_per_cell(TestFixture::GetIFORM));
////
////	x= min-100;
////
////	std::tie(dest,std::ignore)=model.coordinates_global_to_local(x);
////
////	auto range2=model.SelectByNGP( TestFixture::iform, x);
////
////	count =0;
////
////	for(auto s :range2)
////	{
////		EXPECT_EQ( manifold_type::get_cell_index(s),manifold_type::get_cell_index(dest));
////		++count;
////	}
////	EXPECT_EQ(count,manifold_type::get_num_of_comp_per_cell(iform));
//
//}
//}
//
// TYPED_TEST_P(TestModel,SelectByRectangle ){
//{
//
//	auto & geometry= TestFixture::geometry;
//
//	typename TestFixture::coordinate_tuple v0, v1, v2, v3;
//
//	for (int i = 0; i < TestFixture::ndims; ++i)
//	{
//		v0[i] = TestFixture::points[0][i] + TestFixture::dh[i];
//		v1[i] = TestFixture::points[1][i] - TestFixture::dh[i];
//
//		v2[i] = TestFixture::points[0][i] - TestFixture::dh[i] * 2;
//		v3[i] = TestFixture::points[1][i] + TestFixture::dh[i] * 2;
//	}
//	auto domain=make_domain< TestFixture:: GetIFORM>(geometry);
//
//	auto f = make_field<Real>(domain);
//
//
//	f.Clear();
//
//	auto r=geometry.SelectByRectangle( domain, TestFixture::points[0],TestFixture::points[1]);
//
//	auto it= begin(r);
//	auto ie= end(r);
//	for (; it!=ie;++it)
//	{
//		auto s=*it;
//
//		f[s] = 1;
//		auto x = geometry.get_coordinates(s);
//
//		ASSERT_TRUE (
//
//				( (v2[0] - x[0]) * (x[0] - v3[0]) >= 0) &&
//
//				( (v2[1] - x[1]) * (x[1] - v3[1]) >= 0) &&
//
//				( (v2[2] - x[2]) * (x[2] - v3[2]) >= 0)
//
//		)
//
//		;
//	}
////	LOGGER << SAVE(f);
//	for (int i = 0; i < TestFixture::ndims; ++i)
//	{
//		v0[i] = TestFixture::points[0][i] + TestFixture::dh[i];
//		v1[i] = TestFixture::points[1][i] - TestFixture::dh[i];
//
//		v2[i] = TestFixture::points[0][i] - TestFixture::dh[i] * 2;
//		v3[i] = TestFixture::points[1][i] + TestFixture::dh[i] * 2;
//	}
//	for (auto s : domain)
//	{
//		auto x = geometry.get_coordinates(s);
//
//		if (((((v0[0] - x[0]) * (x[0] - v1[0])) >= 0) && (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
//						&& (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0)))
//		{
//			ASSERT_EQ(1,f[s] ) << ( geometry.get_coordinates(s));
//		}
//
//		if (!(((v2[0] - x[0]) * (x[0] - v3[0])) >= 0) && (((v2[1] - x[1]) * (x[1] - v3[1])) >= 0)
//				&& (((v2[2] - x[2]) * (x[2] - v3[2])) >= 0))
//		{
//			ASSERT_NE(1,f[s]) << ( geometry.get_coordinates(s));
//		}
//	}
//
////	LOGGER << SAVE(f);
//
//}}
//
// TYPED_TEST_P(TestModel,SelectByPolylines ){
//{
//
//	auto & geometry= TestFixture::geometry;
//
//	auto domain = make_domain<TestFixture::GetIFORM>(geometry);
//
//	auto f = make_field<Real>(domain);
//
//	f.Clear();
//	f=0;
//	typename TestFixture::coordinate_tuple v0, v1, v2, v3;
//	for (int i = 0; i < TestFixture::ndims; ++i)
//	{
//		v0[i] = TestFixture::points[0][i] + TestFixture::dh[i];
//		v1[i] = TestFixture::points[1][i] - TestFixture::dh[i];
//
//		v2[i] = TestFixture::points[0][i] - TestFixture::dh[i] * 2;
//		v3[i] = TestFixture::points[1][i] + TestFixture::dh[i] * 2;
//	}
//	for (auto s : geometry.SelectByPoints(domain, TestFixture::points))
//	{
//		f[s] = 1;
//		auto x = geometry.coordinates(s);
//
//		ASSERT_TRUE (
//
//				( (v2[0] - x[0]) * (x[0] - v3[0]) >= 0) &&
//
//				( (v2[1] - x[1]) * (x[1] - v3[1]) >= 0) &&
//
//				( (v2[2] - x[2]) * (x[2] - v3[2]) >= 0)
//
//		)
//
//		;
//	}
////	LOGGER << SAVE(f);
//
//}}
//
// TYPED_TEST_P(TestModel,SelectByMaterial ){
//{
//
//	auto & geometry= TestFixture::geometry;
//	static constexpr size_t GetIFORM=TestFixture::GetIFORM;
//
//	auto vertex_domain = make_domain<NODE>(geometry);
//	auto domain = make_domain<TestFixture::GetIFORM>(geometry);
//
//	auto f = make_field<Real>(domain );
//
//	geometry.SetEntity( geometry.SelectByPoints(vertex_domain, TestFixture::points), "Vacuum");
//
//	f.Clear();
//
//	for (auto s : geometry.SelectByMaterial(domain, "Vacuum"))
//	{
//		f[s] = 1;
//	}
////	LOGGER << SAVE(f);
//
//	typename TestFixture::coordinate_tuple v0, v1, v2, v3;
//	for (int i = 0; i < TestFixture::ndims; ++i)
//	{
//		v0[i] = TestFixture::points[0][i] + TestFixture::dh[i];
//		v1[i] = TestFixture::points[1][i] - TestFixture::dh[i];
//
//		v2[i] = TestFixture::points[0][i] - TestFixture::dh[i] * 2;
//		v3[i] = TestFixture::points[1][i] + TestFixture::dh[i] * 2;
//	}
//	for (auto s : domain)
//	{
//		auto x = geometry.coordinates(s);
//
//		if (((((v0[0] - x[0]) * (x[0] - v1[0])) >= 0) && (((v0[1] - x[1]) * (x[1] - v1[1])) >= 0)
//						&& (((v0[2] - x[2]) * (x[2] - v1[2])) >= 0)))
//		{
//			ASSERT_EQ(1,f[s] ) << ( geometry.coordinates(s));
//		}
//
//		if (!(((v2[0] - x[0]) * (x[0] - v3[0])) >= 0) && (((v2[1] - x[1]) * (x[1] - v3[1])) >= 0)
//				&& (((v2[2] - x[2]) * (x[2] - v3[2])) >= 0))
//		{
//			ASSERT_NE(1,f[s]) << ( geometry.coordinates(s));
//		}
//	}
//
//	auto extent = geometry. extents();
//
//	TestFixture::points.emplace_back(typename TestFixture::coordinate_tuple(
//					{	0.3 * extent.m_node_[0], 0.6 * extent.m_node_[1], 0.2 *
// extent.first[2]}));
//
//	geometry.Erase( geometry.SelectByPolylines(vertex_domain, TestFixture::points));
//
//	geometry.SetEntity( geometry.SelectByPolylines(vertex_domain, TestFixture::points), "Plasma");
//
//	for (auto s : geometry.SelectByMaterial( domain, "Plasma"))
//	{
//		f[s] = -1;
//	}
//
//	for (auto s : geometry.SelectInterface( domain, "Plasma", "Vacuum"))
//	{
//		f[s] = 10;
//	}
//
//	for (auto s : geometry.SelectInterface( domain, "Vacuum", "NONE"))
//	{
//		f[s] = -10;
//	}
////	LOGGER << SAVE(f);
//
//}}
//
// REGISTER_TYPED_TEST_CASE_P(TestModel, SelectByNGP, SelectByRectangle,
//		SelectByPolylines, SelectByMaterial);
//
// typedef testing::Types<std::integral_constant<size_t, NODE>,
//		std::integral_constant<size_t, EDGE>,
//		std::integral_constant<size_t, FACE>,
//		std::integral_constant<size_t, CELL> > ParamList;
//
// INSTANTIATE_TYPED_TEST_CASE_P(SimPla, TestModel, ParamList);
