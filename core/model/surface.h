/*
 * surface.h
 *
 *  created on: 2014-4-21
 *      Author: salmon
 */

#ifndef SURFACE_H_
#define SURFACE_H_

#include <stddef.h>
#include <utility>
#include <vector>

#include "../gtl/ntuple.h"
#include "../utilities/primitives.h"
#include "../utilities/type_traits.h"

namespace simpla
{

/**
 *  @ingroup Model
 *
 *
 */

/**
 *  \brief surface
 */

template<typename TM>
class Surface: public std::map<typename TM::index_type, Vec3>
{
public:
	typedef TM mesh_type;
	typedef Surface<TM> this_type;
	typedef std::map<typename TM::index_type, Vec3> base_container_type;

	typedef typename mesh_type::iterator mesh_iterator;
	typedef typename mesh_type::index_type index_type;
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
