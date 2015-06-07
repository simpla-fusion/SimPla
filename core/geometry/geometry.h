/**
 * @file geometry.h
 *
 * @date 2015年6月3日
 * @author salmon
 */

#ifndef CORE_GEOMETRY_GEOMETRY_H_
#define CORE_GEOMETRY_GEOMETRY_H_

#include "../gtl/primitives.h"
#include "../gtl/ntuple.h"

#include "coordinate_system.h"
#include "primitive.h"
#include "chains.h"
//#include "model.h"
#ifdef USE_BOOST
#	error "Custom geometry library is not implemented!"
#else
#	include "boost_gemetry_adapted.h"
#endif

#endif /* CORE_GEOMETRY_GEOMETRY_H_ */
