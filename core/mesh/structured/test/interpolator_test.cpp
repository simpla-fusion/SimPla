/*
 * interpolator_test.cpp
 *
 *  created on: 2014-6-29
 *      Author: salmon
 */

#include "../../structured/test/interpolator_test.h"
#include <tuple>

#include "../../field/field.h"
#include "../../structured/domain.h"
#include "../../structured/topology/structured.h"
#include "../coordinates/coordiantes_cartesian.h"

using namespace simpla;

template<typename TF, template<typename > class TInterpolator> struct TParam
{
	static constexpr size_t iform = field_traits<TF>::iform;
	typedef typename field_traits<TF>::manifold_type manifold_type;
	typedef TInterpolator<manifold_type> interoplator_type;

};
typedef CartesianCoordinates<RectMesh> manifold_type;

typedef testing::Types<
		TParam<Field<Domain<manifold_type, VERTEX>, double>, InterpolatorLinear>

> TypeLists;

INSTANTIATE_TYPED_TEST_CASE_P(SimPla, TestInterpolator, TypeLists);
