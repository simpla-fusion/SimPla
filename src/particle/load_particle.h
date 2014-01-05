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

template<typename > class Particle;

template<typename TConfig, typename TEngine>
bool LoadParticle(TConfig const &cfg, Particle<TEngine> *p)
{

	if (cfg.empty())
	{
		WARNING << "Empty particle config!";

		return false;
	}

	typedef TEngine engine_type;

	typedef typename engine_type::mesh_type mesh_type;

	typedef Particle<engine_type> this_type;

	typedef typename engine_type::Point_s Point_s;

	typedef Point_s value_type;

	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type::coordinates_type coordinate_type;

	mesh_type const &mesh = p->mesh;

	p->engine_type::Deserialize(cfg);

	p->SetName(cfg["Name"].template as<std::string>());

	if (cfg["SRC"].empty()) // Initialize Data
	{

		DEFINE_PHYSICAL_CONST(p->mesh.constants());

		bool doParallel = true;
		size_t pic = cfg["PIC"].template as<size_t>();

		std::function<Real(coordinate_type const & x0)> n;

		if (cfg["n"].empty())
		{
			n = [](coordinate_type const & x0)->Real
			{
				return 1.0;
			};
		}
		else if (cfg["n"].is_number())
		{
			Real n0 = cfg["n"].template as<Real>();

			n = [n0](coordinate_type const & x0)->Real
			{
				return n0;
			};
		}
		else if (cfg["n"].is_function())
		{
			auto l_obj = cfg["n"];

			n = [l_obj](coordinate_type const & x0)->Real
			{
				return l_obj(x0[0],x0[1],x0[2]).template as<Real>();
			};
			doParallel = false;
		}
		else
		{
			Field<Geometry<mesh_type, 0>, Real> n0(mesh);

			LoadField(cfg["n"], &n0);

			n = [n0](coordinate_type const & x0)->Real
			{
				return n0(x0);
			};

		}

		Real vT = 1.0;

		if (!cfg["vT"].empty())
		{
			vT = cfg["vT"].template as<Real>();
		}
		else if (!cfg["T"].empty())
		{
			vT = std::sqrt(2.0 * boltzmann_constant * cfg["T"].template as<Real>() / (p->GetMass()));
		}

		CHECK(vT);

		std::mt19937 rnd_gen(3);

		rectangle_distribution<mesh_type::NUM_OF_DIMS> x_dist;

		multi_normal_distribution<mesh_type::NUM_OF_DIMS> v_dist(vT);

		mesh.SerialTraversal(Particle<TEngine>::IForm,

		[&](typename mesh_type::index_type const & s)
		{

			typename mesh_type::coordinates_type xrange[mesh.GetCellShape(s)];

			mesh.GetCellShape(s,xrange);

			x_dist.Reset(xrange);

			nTuple<3,Real> x,v;

			Real inv_sample_density=1.0/pic;

			for(int i=0;i<pic;++i)
			{
				x_dist(rnd_gen,&x[0]);
				v_dist(rnd_gen,&v[0]);
				p->Insert(s,x,v,
						[&] (coordinate_type const & x0)->Real
						{
							return n(x0)*inv_sample_density;
						}
				);
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
