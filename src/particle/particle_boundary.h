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
template<typename TM>
class ParticleBoundary: public VisitorBase
{
public:

	typedef ParticleBoundary<TM> this_type;

	typedef TM mesh_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;
	typedef typename mesh_type::scalar_type scalar_type;
	typedef nTuple<3, coordinates_type> plane_type;

private:
	typename mesh_type::surface_type surface_;

	std::string op_str_;

	std::function<void(plane_type const&, coordinates_type *, nTuple<3, Real>*)> op_;
public:

	template<typename TDict>
	ParticleBoundary(mesh_type const & mesh, TDict const & dict)
			: op_str_("")
	{
		CreateSurface(mesh, dict["Select"], &surface_);

		if (dict["Operation"].is_string())
		{
			op_str_ = dict["Operation"].template as<std::string>();
		}
		else if (dict["Operation"].is_function())
		{
			auto obj = dict["Operation"];

			op_str_ = "Custom";

//			op_ = [obj](plane_type const& plane, scalar_type f, coordinates_type const& x , nTuple<3, Real> const &v)
//			{
//				obj()
//			}
		}
	}

	virtual ~ParticleBoundary()
	{
	}

	void Visit(void * pp) const
	{

		ParticleBase<TM> & p = *reinterpret_cast<ParticleBase<TM> *>(pp);
		if (op_str_ == "Cycling")
			return;

		LOGGER << "Apply boundary constraint [" << op_str_ << "] to particles [" << p.GetTypeAsString() << "]";

		for (auto const & cell : surface_)
		{
			auto const &plane = cell.second;

			if (op_str_ == "Refelecting")
			{
				p.Modify(cell.first, [&plane](coordinates_type *x, nTuple<3, Real>*v)
				{
					Reflect(plane,x,v);
				});
			}
			else if (op_str_ == "Absorbing")
			{
				p.Remove(cell.first,

				[&](coordinates_type const & x, nTuple<3, Real> const & v)->bool
				{
					return Distance(plane, x )<0;
				}

				);
			}
			else if (op_str_ == "Custom")
			{
				UNIMPLEMENT;
				return;
//				auto foo = [=](scalar_type f,coordinates_type const &x, nTuple<3, Real>const &v)
//				{	op_(f,x,v);};
//				p.Traversal(cell.first, foo);
			}
		}

	}

}
;

}  // namespace simpla

#endif /* PARTICLE_BOUNDARY_H_ */
