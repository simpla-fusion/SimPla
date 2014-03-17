/*
 * load_particle.h
 *
 *  Created on: 2013年12月21日
 *      Author: salmon
 */

#ifndef LOAD_PARTICLE_H_
#define LOAD_PARTICLE_H_

#include <cstddef>
#include <random>
#include <string>
#include <functional>

#include "../fetl/fetl.h"
#include "../fetl/load_field.h"
#include "../numeric/multi_normal_distribution.h"
#include "../numeric/rectangle_distribution.h"
#include "../utilities/log.h"
#include "../physics/physical_constants.h"

namespace simpla
{

template<typename TDict, typename TP, typename TN, typename TT>
bool LoadParticle(TP *p, TDict const &dict, TN const & ne, TT const & Ti)
{

	if (!dict)
	{
		WARNING << "Empty particle configure!";

		return false;
	}

	typedef typename TP::engine_type engine_type;

	typedef typename engine_type::mesh_type mesh_type;

	typedef typename engine_type::Point_s Point_s;

	typedef Point_s value_type;

	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type const &mesh = p->mesh;

	p->engine_type::Load(dict);

	p->SetName(dict["Name"].template as<std::string>());

	if (dict["SRC"].empty()) // Initialize Data
	{

		DEFINE_PHYSICAL_CONST(p->mesh.constants());

		bool doParallel = true;
		size_t pic = dict["PIC"].template as<size_t>();

		std::function<Real(coordinates_type const & x)> n;

		if (ne.empty())
		{
			if (dict["n"].empty())
			{
				ERROR << "Particle density is not defined!!";
			}
			else if (dict["n"].is_number())
			{
				Real n0 = dict["n"].template as<Real>();

				n = [n0](coordinates_type const & x0)->Real
				{
					return n0;
				};
			}
			else if (dict["n"].is_function())
			{
				auto l_obj = dict["n"];

				n = [l_obj](coordinates_type const & x0)->Real
				{
					return l_obj(x0[0],x0[1],x0[2]).template as<Real>();
				};

			}

		}
		else
		{
			if (dict["n"].is_number())
			{
				Real n0 = dict["n"].template as<Real>();

				n = [&](coordinates_type const & x0)->Real
				{
					return n0*ne(x0);
				};
			}
			else
			{
				n = [&](coordinates_type const & x0)->Real
				{
					return ne(x0);
				};
			}
		}

		std::function<Real(coordinates_type const & x)> vT;

		Real a = 2.0 * boltzmann_constant / (p->GetMass());

		if (Ti.empty())
		{
			if (!dict["vT"].empty())
			{
				Real t = dict["vT"].template as<Real>();
				vT = [t](coordinates_type x)
				{	return t;};
			}
			else if (!dict["T"].empty())
			{
				Real t = std::sqrt(a * dict["T"].template as<Real>());

				vT = [t](coordinates_type x)
				{	return t;};
			}
			else
			{
				ERROR << " Particle temperature is not defined!!";
			}
		}
		else
		{
			vT = [&](coordinates_type x)->Real
			{	return std::sqrt(a * Ti(x));};

		}

		std::mt19937 rnd_gen(3);

		rectangle_distribution<mesh_type::NDIMS> x_dist;

		multi_normal_distribution<mesh_type::NDIMS> v_dist;

		mesh.template Traversal<TP::IForm>(

		[&](typename mesh_type::index_type s)
		{

			nTuple<3,Real> x,v;

			Real inv_sample_density=1.0/pic;

			for(int i=0;i<pic;++i)
			{
				x_dist(rnd_gen,&x[0]);
				v_dist(rnd_gen,&v[0]);

				x=mesh.CoordinatesLocalToGlobal(s,x);
				v=mesh.PushForward(x,v) * vT(x);
				p->Insert(s, engine_type::make_point(x, v,n(x)*inv_sample_density ));
			}
		});
	}
	else // read data from file
	{
		UNIMPLEMENT2("Read  particle data from file");
	}

	LOGGER

	<< "Load Particle:[ Name=" << p->GetName()

	<< ", Engine=" << p->GetTypeAsString()

	<< ", Number of Particles=" << p->size() << "]";

	return true;
}

}  // namespace simpla

#endif /* LOAD_PARTICLE_H_ */
