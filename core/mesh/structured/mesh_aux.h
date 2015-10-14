/**
 * @file mesh_aux.h.h
 * @author salmon
 * @date 2015-10-14.
 */

#ifndef SIMPLA_MESH_AUX_H_H
#define SIMPLA_MESH_AUX_H_H

#include "../../gtl/utilities/log.h"

namespace simpla
{
namespace policy
{

template<typename ...> class MeshUtilities;


template<typename TGeo>
struct MeshUtilities<TGeo>
{
	typedef TGeo geometry_type;

	MeshUtilities()
	{

	}

	virtual ~MeshUtilities() { }

	virtual geometry_type const &geometry() const = 0;

	virtual void update()
	{
		LOGGER << geometry().get_type_as_string() << std::endl;
	}
};

}//namespace policy

}//namespace simpla
#endif //SIMPLA_MESH_AUX_H_H
