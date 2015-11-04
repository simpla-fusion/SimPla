/**
 * @file geometry.h
 *
 * @date 2015-6-3
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
#	error "Custom geometry_obj library is not implemented!"
#else

#	include "boost_gemetry_adapted.h"

#endif

namespace simpla
{


//template<typename, typename> struct map;

template<typename> struct Mertic;

}//namespace simpla

#endif /* CORE_GEOMETRY_GEOMETRY_H_ */
