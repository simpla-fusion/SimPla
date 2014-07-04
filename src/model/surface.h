/*
 * surface.h
 *
 *  Created on: 2014年4月21日
 *      Author: salmon
 */

#ifndef SURFACE_H_
#define SURFACE_H_

#include <stddef.h>
#include <utility>
#include <vector>

#include "../utilities/ntuple.h"
#include "../utilities/primitives.h"
#include "../utilities/sp_type_traits.h"

namespace simpla
{

/**
 *  @ingroup Model Geometry
 *
 *
 */

/**
 *  @brief surface
 */

template<typename TM>
class Surface: public std::map<typename TM::compact_index_type, Vec3>
{
public:
	typedef TM mesh_type;
	typedef Surface<TM> this_type
	typedef std::map<typename TM::compact_index_type, Vec3> base_container_type;

	typedef typename mesh_type::iterator mesh_iterator;
	typedef typename mesh_type::compact_index_type compact_index_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	Surface()
	{
	}
	~Surface()
	{
	}
};

}
// namespace simpla

#endif /* SURFACE_H_ */
