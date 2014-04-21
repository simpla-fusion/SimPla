/*
 * particle_constraint.h
 *
 *  Created on: 2014年4月21日
 *      Author: salmon
 */

#ifndef PARTICLE_CONSTRAINT_H_
#define PARTICLE_CONSTRAINT_H_
#include "material.h"
#include "surface.h"
namespace simpla
{
template<typename TM, typename TDict>
void CreateSurface(Surface<TM> *surf, TDict const &dict)
{
	if (dict["Width"].is_number())
	{
		CreateSurface(surf, dict["Width"].template as<Real>());
	}
	else
	{
		WARNING << "Unknown configuation!";
	}
}
template<typename TM>
void CreateSurface(Surface<TM> *surf, Real width)
{
	TM const & mesh = surf->mesh;

	auto extent = mesh.GetExtent();
	auto dims = mesh.GetDimensions();
	auto xmin = extent.first;
	auto xmax = extent.second;
	auto d = mesh.GetDx();
	nTuple<3, Real> dx = { 0.5 * d[0], 0, 0 };
	nTuple<3, Real> dy = { 0, 0.5 * d[0], 0 };
	nTuple<3, Real> dz = { 0, 0, 0.5 * d[0] };

	nTuple<3, Real> idx = { -0.5 * d[0], 0, 0 };
	nTuple<3, Real> idy = { 0, -0.5 * d[0], 0 };
	nTuple<3, Real> idz = { 0, 0, -0.5 * d[0] };

	for (auto s : mesh.GetRange(VERTEX))
	{
		auto x = mesh.GetCoordinates(s);

		if (x[0] < xmin[0] + width)
		{
			surf->insert(s, dx);
			continue;
		}
		else if (x[0] > xmax[0] - width)
		{
			surf->insert(s, idx);
			continue;
		}

		if (x[1] < xmin[1] + width)
		{
			surf->insert(s, dy);
			continue;
		}
		else if (x[1] > xmax[1] + width)
		{
			surf->insert(s, idy);
			continue;
		}

		if (x[2] < xmin[2] + width)
		{
			surf->insert(s, dz);
			continue;
		}
		else if (x[2] > xmax[2] - width)
		{
			surf->insert(s, idz);
			continue;
		}

	}
}

template<typename TM, typename TDict>
std::function<void(std::string const &, std::shared_ptr<ParticleBase<TM>>)> CreateParticleConstraint(
        Material<TM> const & material, TDict const & dict)
{
	std::function<void(std::string const &, std::shared_ptr<ParticleBase<TM>>)> res =
	        [](std::string const &, std::shared_ptr<ParticleBase<TM>>)
	        {
		        WARNING<<"Nothing to do!";
	        };

	typedef TM mesh_type;

	mesh_type const & mesh = material.mesh;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	std::shared_ptr<Surface<TM>> surface(new Surface<TM>(mesh));

	auto particle_name = dict["Name"].template as<std::string>("All");

	auto op_str = dict["Type"].template as<std::string>("Absorb");

	CreateSurface(surface.get(), dict["Select"]);

	return [=](std::string const & name, std::shared_ptr<ParticleBase<TM>> p)
	{
		if(particle_name=="All"|| particle_name==name)
		{
			LOGGER<< "Particle " << particle_name<<" is  "<< op_str<<"ed on the boundary";
			p->Boundary(*surface,op_str);
		}
	};;
}

}  // namespace simpla

#endif /* PARTICLE_CONSTRAINT_H_ */
