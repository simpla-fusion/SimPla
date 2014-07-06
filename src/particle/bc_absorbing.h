/*
 * bc_absorbing.h
 *
 *  Created on: 2014-4-24
 *      Author: salmon
 */

#ifndef BC_ABSORBING_H_
#define BC_ABSORBING_H_

#include <memory>
#include <string>

#include "../utilities/ntuple.h"
#include "../utilities/primitives.h"
#include "particle_boundary.h"

namespace simpla
{
template<typename TP>
class AbsorbingBoundary: public ParticleBoundary<TP>
{

public:
	typedef AbsorbingBoundary<TP> this_type;
	typedef ParticleBoundary<TP> base_type;

	template<typename TDict>
	AbsorbingBoundary(mesh_type const & mesh, TDict const & dict)
			: base_type(mesh, dict)
	{
	}

	~AbsorbingBoundary()
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
		return "Absorbing";
	}
	std::string GetTypeAsString_() const
	{
		return GetTypeAsString();
	}
	void Visit(void * pp) const
	{
		LOGGER << "Apply boundary constraint [" << GetTypeAsString() << "] to particles [" << TP::GetTypeAsString()
		        << "]";

		particle_type & p = *reinterpret_cast<particle_type*>(pp);

		for (auto const &s : surface_)
		{
			coordinates_type x;
			nTuple<3, Real> v;
			for (auto & point : p[s.first])
			{
				p.PullBack(point, &x, &v);
				Relection(s.second, &x, &v);
				p.PushForward(x, v, &point);
			}

			p.Resort(s.first);
		}

	}
};
}  // namespace simpla

#endif /* BC_ABSORBING_H_ */
