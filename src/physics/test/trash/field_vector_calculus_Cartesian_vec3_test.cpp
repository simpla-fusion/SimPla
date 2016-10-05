/*
 * fetl_test.cpp
 *
 *  created on: 2013-12-28
 *      Author: salmon
 */

#include "../../diff_geometry/diff_scheme/fdm.h"
#include "../../diff_geometry/geometry/cartesian.h"
#include "../../diff_geometry/interpolator/interpolator.h"
#include "../../diff_geometry/mesh.h"
#include "../../diff_geometry/topology/structured.h"
#include "../../toolbox/ntuple.h"

using namespace simpla;

typedef Manifold<CartesianCoordinate<RectMesh, CARTESIAN_ZAXIS>,
		FiniteDiffMethod, InterpolatorLinear> m_type;

typedef nTuple<Real, 3> v_type;

#include "field_diff_calculus_test.h"

