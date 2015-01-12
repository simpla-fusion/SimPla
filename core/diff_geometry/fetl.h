/**
 * Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id: FETL.h 1009 2011-02-07 23:20:45Z salmon $
 * @file fetl.h
 *
 *  created on: 2009-3-31
 *      Author: salmon
 */
#ifndef FETL_H_
#define FETL_H_

#include "../diff_geometry/calculus.h"
#include "../diff_geometry/diff_scheme/fdm.h"
#include "../diff_geometry/domain.h"
#include "../diff_geometry/geometry/cartesian.h"
#include "../diff_geometry/interpolator/interpolator.h"
#include "../diff_geometry/manifold.h"
#include "../diff_geometry/topology/structured.h"
#include "../utilities/utilities.h"
#include "../field/field.h"


namespace simpla
{
typedef Manifold<CartesianCoordinates<StructuredMesh>, FiniteDiffMethod,
		InterpolatorLinear> CartesianMesh;

//typedef Manifold<CylindricalCoordinates<StructuredMesh>, FiniteDiffMethod,
//		InterpolatorLinear> CylindricalMesh;
}  // namespace simpla

#endif  // FETL_H_
