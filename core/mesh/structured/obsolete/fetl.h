/**
 * Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id: FETL.h 1009 2011-02-07 23:20:45Z salmon $
 *
 * @file fetl.h
 *
 *  created on: 2009-3-31
 *      Author: salmon
 */
#ifndef FETL_H_
#define FETL_H_

#include "../diff_geometry/manifold.h"
#include "../utilities/utilities.h"
#include "../field/field.h"
#include "../structured/calculus.h"
#include "../structured/diff_scheme/fdm.h"
#include "../structured/domain.h"
#include "../structured/interpolator/interpolator.h"
#include "../structured/topology/structured.h"
#include "coordinates/coordiantes_cartesian.h"


namespace simpla
{
typedef Manifold<CartesianCoordinates<RectMesh>, FiniteDiffMethod,
		InterpolatorLinear> CartesianMesh;

//typedef Manifold<CylindricalCoordinates<StructuredMesh>, FiniteDiffMethod,
//		InterpolatorLinear> CylindricalMesh;
}  // namespace simpla

#endif  // FETL_H_
