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

template<typename TP, typename TDict, typename ...Args>
void LoadParticle(TP *p, TDict const &dict, Args const & ... args)
{
	if (!dict)
	{
		WARNING << "Empty particle configure!";
		return;
	}
	else if (!dict["URL"].empty()) // Load Data from file
	{
		UNIMPLEMENT2("Read  particle data from file");
		return;
	}

	p->J.Clear();
	p->n.Clear();

	InitParticle(p, dict, std::forward<Args const &>(args)...);

	LOGGER << "Create Particles:[ Engine=" << p->GetTypeAsString() << ", Number of Particles=" << p->size() << "]";

	LOGGER << DONE;
}

template<typename TP, typename TDict>
void InitParticle(TP *p, TDict const &dict)
{
	typedef typename TP::mesh_type::coordinates_type coordinates_type;

	std::function<Real(coordinates_type const &)> ns, Ts;

	if (dict["Density"].is_number())
	{
		Real n0 = dict["Density"].template as<Real>();

		ns = [n0](coordinates_type )->Real
		{
			return n0;
		};
	}
	else if (dict["Density"].is_function())
	{
		auto l_obj = dict["Density"];

		ns = [l_obj](coordinates_type x)->Real
		{
			return l_obj(x[0],x[1],x[2]).template as<Real>();
		};

	}
	else
	{
		ERROR << "Particle density is not defined!";
	}

	if (dict["Temperature"].is_number())
	{
		Real T = dict["Temperature"].template as<Real>();

		Ts = [T](coordinates_type const & )->Real
		{
			return T;
		};
	}
	else if (dict["Temperature"].is_function())
	{
		auto l_obj = dict["Temperature"];

		Ts = [l_obj](coordinates_type const & x )->Real
		{
			return l_obj(x[0],x[1],x[2]).template as<Real>();
		};

	}
	else
	{
		ERROR << "Particle temperature is not defined!";
	}

	InitParticle(p, p->mesh.GetRange(TP::IForm), dict["PIC"].template as<size_t>(100), ns, Ts);

}

template<typename TDict, typename TP, typename TN, typename TT>
void InitParticle(TP *p, TDict const &dict, TN const & ne, TT const & Ti)
{
	if (ne.empty() || Ti.empty())
	{
		InitParticle(p, dict);
		return;
	}

	typedef typename TP::mesh_type::coordinates_type coordinates_type;

	Real n0 = dict["Proportion"].template as<Real>(1.0);

	InitParticle(p, p->mesh.GetRange(TP::IForm), dict["PIC"].template as<size_t>(100),

	[&](coordinates_type x)->Real
	{	return n0*ne(x);},

	Ti);

}

template<typename TP, typename TR, typename TN, typename TT>
void InitParticle(TP *p, TR range, size_t pic, TN const & ns, TT const & Ts)
{
	typedef typename TP::engine_type engine_type;

	typedef typename TP::mesh_type mesh_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	static constexpr int NDIMS = mesh_type::NDIMS;

	mesh_type const &mesh = p->mesh;

	DEFINE_PHYSICAL_CONST(p->mesh.constants());

	nTuple<NDIMS, Real> dxmin = { -0.5, -0.5, -0.5 };
	nTuple<NDIMS, Real> dxmax = { 0.5, 0.5, 0.5 };
	rectangle_distribution<NDIMS> x_dist(dxmin, dxmax);
	multi_normal_distribution<NDIMS> v_dist;

	std::mt19937 rnd_gen(NDIMS * 2);

	nTuple<3, Real> x, v;

	for (auto s : range)
	{

		Real inv_sample_density = p->GetCharge() * mesh.Volume(s) / pic;

		p->n[s] = mesh.Sample(Int2Type<TP::IForm>(), s, p->GetCharge() * ns(mesh.GetCoordinates(s)));

		for (int i = 0; i < pic; ++i)
		{
			x_dist(rnd_gen, &x[0]);

			v_dist(rnd_gen, &v[0]);

			x = mesh.CoordinatesLocalToGlobal(s, x);

			v = mesh.PushForward(x, v) * std::sqrt(boltzmann_constant * Ts(x) / p->GetMass());

			p->Insert(s, engine_type::make_point(x, v, ns(x) * inv_sample_density));
		}

	}
}
}  // namespace simpla

#endif /* LOAD_PARTICLE_H_ */
