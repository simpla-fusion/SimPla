/*
 * fetl_test.cpp
 *
 *  created on: 2013-12-28
 *      Author: salmon
 */

#include "../../utilities/ntuple.h"
#include "../../manifold/manifold.h"
#include "../../manifold/geometry/cartesian.h"
#include "../../manifold/topology/structured.h"
#include "../../manifold/diff_scheme/fdm.h"
#include "../../manifold/interpolator/interpolator.h"
#include "../field.h"

using namespace simpla;

typedef Manifold<CartesianCoordinates<StructuredMesh, CARTESIAN_ZAXIS>,
		FiniteDiffMethod, InterpolatorLinear> m_type;
typedef Real v_type;

#include "field_vector_calculus_test.h"

