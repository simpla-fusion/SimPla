/*
 * mesh_test.cpp
 *
 *  Created on: 2013年12月29日
 *      Author: salmon
 */

#include <gtest/gtest.h>

#include "../fetl/fetl.h"
#include "../io/data_stream.h"
#include "../utilities/log.h"
#include "../utilities/lua_state.h"

#include "co_rect_mesh.h"
#include "media_tag.h"
using namespace simpla;

typedef testing::Types<CoRectMesh<Real>
//, CoRectMesh<Complex>
> AllMeshTypes;

template<typename TMesh>
class TestMesh: public testing::Test
{
protected:
	virtual void SetUp()
	{
		Logger::Verbose(10);
	}
public:

	typedef TMesh mesh_type;

	DEFINE_FIELDS(mesh_type)
};

TYPED_TEST_CASE(TestMesh, AllMeshTypes);

TYPED_TEST(TestMesh,create_default){
{
	typedef typename TestFixture::mesh_type mesh_type;

	mesh_type mesh;

	mesh.Update();

	LOGGER<<mesh;

}
}
TYPED_TEST(TestMesh,Create_parse_cfg){
{
	LuaObject cfg;
	cfg.ParseString(

			" Grid=                                                                       \n"
			" {                                                                                            \n"
			"   Type=\"CoRectMesh\",                                                                       \n"
			"   ScalarType=\"Real\",                                                                  \n"
			"   UnitSystem={Type=\"SI\"},                                                                  \n"
			"   Topology=                                                                                  \n"
			"   {                                                                                          \n"
			"       Type=\"3DCoRectMesh\",                                                                 \n"
			"       Dimensions={100,100,100}, -- number of grid, now only first dimension is valid            \n"
			"       GhostWidth= {5,0,0},  -- width of ghost points  , if gw=0, coordinate is               \n"
			"                               -- Periodic at this direction                                  \n"
			"   },                                                                                         \n"
			"   Geometry=                                                                                  \n"
			"   {                                                                                          \n"
			"       Type=\"Origin_DxDyDz\",                                                                \n"
			"       Min={0.0,0.0,0.0},                                                                     \n"
			"       Max={1.0,1.0,1.0},                                                                     \n"
			"       dt=0.5*1.0/ (100.0-1.0)  -- time step                                                      \n"
			"   }                                                                                          \n"
			"}                                                                                             \n"

	);

	typedef typename TestFixture::mesh_type mesh_type;

	mesh_type mesh;

	mesh.Deserialize( cfg["Grid"]);

	LOGGER<<mesh;

}
}

template<typename TMesh>
class TestMeshFunctions: public testing::Test
{
protected:
	virtual void SetUp()
	{
		Logger::Verbose(10);

		mesh.dt_ = 1.0;
		mesh.xmin_[0] = 0;
		mesh.xmin_[1] = 0;
		mesh.xmin_[2] = 0;
		mesh.xmax_[0] = 1.0;
		mesh.xmax_[1] = 1.0;
		mesh.xmax_[2] = 1.0;
		mesh.dims_[0] = 20;
		mesh.dims_[1] = 10;
		mesh.dims_[2] = 10;
		mesh.dt_ = 1.0;

		mesh.Update();

		GLOBAL_DATA_STREAM.OpenFile("");

	}
public:

	typedef TMesh mesh_type;

	DEFINE_FIELDS(mesh_type)

	mesh_type mesh;

};

TYPED_TEST_CASE(TestMeshFunctions, AllMeshTypes);

TYPED_TEST(TestMeshFunctions,add_tags){
{
	typedef typename TestFixture::mesh_type mesh_type;

	mesh_type & mesh=TestFixture::mesh;

	LuaObject cfg;

	cfg.ParseString(
			" Media=                                                                 "
			" {                                                                      "
			"    {Type=\"Vacuum\",Region={{0.2,0,0},{0.8,0,0}},Op=\"Set\"},          "
			"                                                                        "
			"    {Type=\"Plasma\",                                                   "
			"      Select=function(x,y,z)                                            "
			"           return x>1.0 and x<2.0                                       "
			"         end                                                            "
			"     ,Op=\"Set\"},                                                      "
			" }                                                                      "
	);

	mesh.tags().Deserialize(cfg["Media"]);

}}
