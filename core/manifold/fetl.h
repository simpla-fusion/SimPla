/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id: FETL.h 1009 2011-02-07 23:20:45Z salmon $
 * FETL.h
 *
 *  created on: 2009-3-31
 *      Author: salmon
 */
#ifndef FETL_H_
#define FETL_H_

#include "../utilities/utilities.h"
#include "../field/field.h"

#include "manifold.h"
#include "domain.h"
#include "calculus.h"

#include "geometry/cartesian.h"
//#include "geometry/cylindrical.h"
#include "topology/structured.h"
#include "diff_scheme/fdm.h"
#include "interpolator/interpolator.h"

namespace simpla
{
typedef Manifold<CartesianCoordinates<StructuredMesh>, FiniteDiffMethod,
		InterpolatorLinear> CartesianMesh;

//typedef Manifold<CylindricalCoordinates<StructuredMesh>, FiniteDiffMethod,
//		InterpolatorLinear> CylindricalMesh;
}  // namespace simpla

#endif  // FETL_H_
