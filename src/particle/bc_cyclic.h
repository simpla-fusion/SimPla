/*
 * bc_cyclic.h
 *
 *  Created on: 2014-4-24
 *      Author: salmon
 */

#ifndef BC_CYCLIC_H_
#define BC_CYCLIC_H_
#include <memory>
#include <string>

#include "../utilities/ntuple.h"
#include "../utilities/primitives.h"
#include "particle_boundary.h"

namespace simpla
{
template<typename TP>
class CyclicBoundary: public ParticleBoundary<TP>
{

public:
	typedef AbsorbingBoundary<TP> this_type;
	typedef ParticleBoundary<TP> base_type;

	template<typename TDict>
	CyclicBoundary(mesh_type const & mesh, TDict const & dict)
			: base_type(mesh, dict)
	{
	}

	~CyclicBoundary()
	{
	}

	template<typename TDict>
	static std::shared_ptr<this_type> Create(mesh_type const & mesh, TDict const & dict)
	{
		std::shared_ptr<this_type> res(nullptr);

		if (dict["Type"].template as<std::string>("Unknown") == GetTypeAsString())
		{
			res = std::shared_ptr<this_type>(new this_type(mesh, dict));
		}

		return res;
	}

	static std::string GetTypeAsString()
	{
		return "Cyclic";
	}
	std::string GetTypeAsString_() const
	{
		return GetTypeAsString();
	}
	void Visit(void * pp) const
	{
		LOGGER << "Apply boundary constraint [" << GetTypeAsString() << "] to particles [" << TP::GetTypeAsString()
		        << "]";

		VERBOSE << "Cyclic boundary is the default boundary condition.";
	}
};
}  // namespace simpla

#endif /* BC_CYCLIC_H_ */
