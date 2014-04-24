/*
 * particle_boundary.h
 *
 *  Created on: 2014年4月24日
 *      Author: salmon
 */

#ifndef PARTICLE_BOUNDARY_H_
#define PARTICLE_BOUNDARY_H_

#include <map>

#include "../fetl/ntuple.h"
#include "../utilities/visitor.h"
#include "../modeling/geometry_algorithm.h"

namespace simpla
{
template<typename TP>
class ParticleBoundary: public VisitorBase
{
public:

	static constexpr unsigned int IForm = TP::IForm;

	typedef TP particle_type;

	typedef ParticleBoundary<particle_type> this_type;

	typedef typename particle_type::mesh_type mesh_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename particle_type::value_type value_type;

	typedef nTuple<3, coordinates_type> plane_type;

private:
	std::map<index_type, plane_type> surface_;
public:

	template<typename TDict>
	ParticleBoundary(mesh_type const & mesh, TDict const & dict)
	{
		CreateSurface(mesh, dict["Select"], &surface_);
	}

	virtual ~ParticleBoundary()
	{
	}

	virtual void Visit(void * pp) const=0


}
;

}  // namespace simpla

#endif /* PARTICLE_BOUNDARY_H_ */
