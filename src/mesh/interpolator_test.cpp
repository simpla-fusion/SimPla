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

TParam<Mesh<CartesianGeometry<UniformArray>, false>, VERTEX> //
        , TParam<Mesh<CartesianGeometry<UniformArray>, false>, EDGE> //
        , TParam<Mesh<CartesianGeometry<UniformArray>, false>, FACE> //
        , TParam<Mesh<CartesianGeometry<UniformArray>, false>, VOLUME> //

        , TParam<Mesh<CartesianGeometry<UniformArray>, true>, VERTEX> //
        , TParam<Mesh<CartesianGeometry<UniformArray>, true>, EDGE> //
        , TParam<Mesh<CartesianGeometry<UniformArray>, true>, FACE> //
        , TParam<Mesh<CartesianGeometry<UniformArray>, true>, VOLUME> //

        , TParam<Mesh<CylindricalGeometry<UniformArray>, false>, VERTEX> //
        , TParam<Mesh<CylindricalGeometry<UniformArray>, false>, EDGE> //
        , TParam<Mesh<CylindricalGeometry<UniformArray>, false>, FACE> //
        , TParam<Mesh<CylindricalGeometry<UniformArray>, false>, VOLUME> //

        , TParam<Mesh<CylindricalGeometry<UniformArray>, true>, VERTEX> //
        , TParam<Mesh<CylindricalGeometry<UniformArray>, true>, EDGE> //
        , TParam<Mesh<CylindricalGeometry<UniformArray>, true>, FACE> //
        , TParam<Mesh<CylindricalGeometry<UniformArray>, true>, VOLUME> //
//
//,Mesh<CartesianGeometry<UniformArray>, true>    //
//,Mesh<CylindricalGeometry<UniformArray>, true>   //
//,Mesh<CylindricalGeometry<UniformArray>, false>  //

> MeshTypeList;

INSTANTIATE_TYPED_TEST_CASE_P(SimPla, TestInterpolator, MeshTypeList);
