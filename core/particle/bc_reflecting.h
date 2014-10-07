/*
 * bc_reflecting.h
 *
 *  created on: 2014-4-24
 *      Author: salmon
 */

#ifndef BC_REFLECTING_H_
#define BC_REFLECTING_H_

#include <memory>
#include <string>

#include "../utilities/ntuple.h"
#include "../utilities/primitives.h"
#include "../utilities/log.h"
#include "particle_boundary.h"

namespace simpla
{
template<typename TM>
class ReflectingBoundary: public ParticleBoundary<TM>
{

public:
	typedef ReflectingBoundary<TP> this_type;
	typedef ParticleBoundary<TP> base_type;

	template<typename TDict>
	ReflectingBoundary(mesh_type const & mesh, TDict const & dict)
			: base_type(mesh, dict)
	{
	}

	~ReflectingBoundary()
	{
	}

	template<typename TDict>
	static std::shared_ptr<this_type> create(mesh_type const & mesh, TDict const & dict)
	{
		std::shared_ptr<this_type> res(nullptr);

		if (dict["Type"].template as<std::string>("Unknown") == get_type_as_string())
		{
			res = std::shared_ptr<this_type>(new this_type(mesh, dict));
		}

		return res;
	}

	static std::string get_type_as_string()
	{
		return "Reflecting";
	}
	std::string get_type_as_string() const
	{
		return get_type_as_string();
	}
	void Visit(void * pp) const
	{
		LOGGER << "Apply boundary constraint [" << get_type_as_string() << "] to particles [" << TP::get_type_as_string()
		        << "]";

		particle_type & p = *reinterpret_cast<particle_type*>(pp);

		for (auto const &s : surface_)
		{
			coordinates_type x;
			nTuple<3, Real> v;
			for (auto & point : p[s.first])
			{
				p.pull_back(pounsigned int , &x, &v);
				Relection(s.second, &x, &v);
				p.push_forward(x, v, &point);
			}

			p.Resort(s.first);
		}

	}
};
}  // namespace simpla

#endif /* BC_REFLECTING_H_ */
