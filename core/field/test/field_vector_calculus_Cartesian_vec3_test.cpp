/*
 * fetl_test.cpp
 *
 *  created on: 2013-12-28
 *      Author: salmon
 */

#include "../../diff_geometry/diff_scheme/fdm.h"
#include "../../diff_geometry/geometry/cartesian.h"
#include "../../diff_geometry/interpolator/interpolator.h"
#include "../../diff_geometry/manifold.h"
#include "../../diff_geometry/topology/structured.h"
#include "../../utilities/ntuple.h"

using namespace simpla;

typedef Manifold<CartesianCoordinates<StructuredMesh, CARTESIAN_ZAXIS>,
		FiniteDiffMethod, InterpolatorLinear> m_type;

typedef nTuple<Real, 3> v_type;

#include "field_vector_calculus_test.h"

