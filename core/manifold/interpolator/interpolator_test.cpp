/*
 * interpolator_test.cpp
 *
 *  created on: 2014-6-29
 *      Author: salmon
 */

#include "interpolator_test.h"

#include "mesh_rectangle.h"
#include "geometry_cartesian.h"
#include "geometry_cylindrical.h"
#include "uniform_array.h"

template<typename TM,  unsigned int  IFORM> struct TParam
{
	typedef TM mesh_type;
	static constexpr   unsigned int   IForm = IFORM;
};
typedef ::testing::Types<

TParam<Mesh<CartesianCoordinates<SurturedMesh>, false>, VERTEX> //
        , TParam<Mesh<CartesianCoordinates<SurturedMesh>, false>, EDGE> //
        , TParam<Mesh<CartesianCoordinates<SurturedMesh>, false>, FACE> //
        , TParam<Mesh<CartesianCoordinates<SurturedMesh>, false>, VOLUME> //

        , TParam<Mesh<CartesianCoordinates<SurturedMesh>, true>, VERTEX> //
        , TParam<Mesh<CartesianCoordinates<SurturedMesh>, true>, EDGE> //
        , TParam<Mesh<CartesianCoordinates<SurturedMesh>, true>, FACE> //
        , TParam<Mesh<CartesianCoordinates<SurturedMesh>, true>, VOLUME> //

        , TParam<Mesh<CylindricalGeometry<SurturedMesh>, false>, VERTEX> //
        , TParam<Mesh<CylindricalGeometry<SurturedMesh>, false>, EDGE> //
        , TParam<Mesh<CylindricalGeometry<SurturedMesh>, false>, FACE> //
        , TParam<Mesh<CylindricalGeometry<SurturedMesh>, false>, VOLUME> //

        , TParam<Mesh<CylindricalGeometry<SurturedMesh>, true>, VERTEX> //
        , TParam<Mesh<CylindricalGeometry<SurturedMesh>, true>, EDGE> //
        , TParam<Mesh<CylindricalGeometry<SurturedMesh>, true>, FACE> //
        , TParam<Mesh<CylindricalGeometry<SurturedMesh>, true>, VOLUME> //
//
//,Mesh<CartesianGeometry<UniformArray>, true>    //
//,Mesh<CylindricalGeometry<UniformArray>, true>   //
//,Mesh<CylindricalGeometry<UniformArray>, false>  //

> MeshTypeList;

INSTANTIATE_TYPED_TEST_CASE_P(SimPla, TestInterpolator, MeshTypeList);
