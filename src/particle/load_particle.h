/*
 * load_particle.h
 *
 *  created on: 2013-12-21
 *      Author: salmon
 */

#ifndef LOAD_PARTICLE_H_
#define LOAD_PARTICLE_H_

#include <random>
#include <string>
#include <functional>

#include "../fetl/fetl.h"
#include "../fetl/load_field.h"

#include "../numeric/multi_normal_distribution.h"
#include "../numeric/rectangle_distribution.h"

#include "../physics/physical_constants.h"

#include "../particle/particle_base.h"

#include "../utilities/log.h"
#include "../utilities/utilities.h"
#include "../parallel/mpi_aux_functions.h"

namespace simpla
{

template<typename TP, typename TDict, typename TModel, typename TN, typename TT>
std::shared_ptr<ParticleBase> LoadParticle(TDict const &dict, TModel const & model, TN const & ne0, TT const & T0)
{
	if (!dict || (TP::get_type_as_string_static() != dict["Type"].template as<std::string>()))
	{
		PARSER_ERROR("illegal particle configure!");
	}

	typedef typename TP::mesh_type mesh_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	std::shared_ptr<TP> res(new TP(dict, model));

	std::function<Real(coordinates_type const&)> ns;

	std::function<Real(coordinates_type const&)> Ts;

	if (!T0.empty())
	{
		Ts = [&T0](coordinates_type x)->Real
		{	return T0(x);};
	}
	else if (dict["Temperature"].is_number())
	{
		Real T = dict["Temperature"].template as<Real>();
		Ts = [T](coordinates_type x)->Real
		{	return T;};
	}
	else if (dict["Temperature"].is_function())
	{
		Ts = dict["Temperature"].template as<std::function<Real(coordinates_type const&)>>();
	}

	if (!ne0.empty())
	{
		Real ratio = dict["Ratio"].template as<Real>(1.0);
		ns = [&ne0,ratio](coordinates_type x)->Real
		{	return ne0(x)*ratio;};
	}
	else if (dict["Density"].is_number())
	{
		Real n0 = dict["Density"].template as<Real>();
		ns = [n0](coordinates_type x)->Real
		{	return n0;};
	}
	else if (dict["Density"].is_function())
	{
		ns = dict["Density"].template as<std::function<Real(coordinates_type const&)>>();
	}

	unsigned int pic = dict["PIC"].template as<size_t>(100);

	auto range = model.SelectByConfig(TP::IForm, dict["Select"]);

	InitParticle(res.get(), range, pic, ns, Ts);

	LoadParticleConstriant(res.get(), range, model, dict["Constraints"]);

	LOGGER << "Create Particles:[ Engine=" << res->get_type_as_string() << ", Number of Particles=" << res->size()
	        << "]" << DONE;

	return std::dynamic_pointer_cast<ParticleBase>(res);

}

template<typename TP, typename TRange, typename TModel, typename TDict>
void LoadParticleConstriant(TP *p, TRange const &range, TModel const & model, TDict const & dict)
{
	if (!dict)
		return;

	for (auto const & key_item : dict)
	{
		auto const & item = std::get<1>(key_item);

		auto r = model.SelectByConfig(range, item["Select"]);

		auto type = item["Type"].template as<std::string>("Modify");

		if (type == "Modify")
		{
			p->AddConstraint([=]()
			{	p->Modify(r, item["Operations"]);});
		}
		else if (type == "Remove")
		{
			if (item["Operation"])
			{
				p->AddConstraint([=]()
				{	p->Remove(r);});
			}
			else if (item["Condition"])
			{
				p->AddConstraint([=]()
				{	p->Remove(r,item["Condition"]);});
			}
		}

	}
}

template<typename TP, typename TR, typename TN, typename TT>
void InitParticle(TP *p, TR range, size_t pic, TN const & ns, TT const & Ts)
{
	typedef typename TP::engine_type engine_type;

	typedef typename TP::mesh_type mesh_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	static constexpr unsigned int NDIMS = mesh_type::NDIMS;

	mesh_type const &mesh = p->mesh;

	DEFINE_PHYSICAL_CONST

	std::mt19937 rnd_gen(NDIMS * 2);

	size_t number = size_of_range(range);

	std::tie(number, std::ignore) = sync_global_location(number * pic * NDIMS * 2);

	rnd_gen.discard(number);

	nTuple<3, Real> x, v;

	Real inv_sample_density = 1.0 / pic;

	auto buffer = p->create_child();

	rectangle_distribution<NDIMS> x_dist;

	multi_normal_distribution<NDIMS> v_dist;

	for (auto s : range)
	{

		for (int i = 0; i < pic; ++i)
		{
			x_dist(rnd_gen, &x[0]);

			v_dist(rnd_gen, &v[0]);

			x = mesh.CoordinatesLocalToGlobal(s, x);

			v *= std::sqrt(boltzmann_constant * Ts(x) / p->m);

			buffer.push_back(engine_type::make_point(x, v, ns(x) * inv_sample_density));
		}

		auto & d = p->get(s);
		d.splice(d.begin(), buffer);
	}

	p->Add(&buffer);
	update_ghosts(p);

}
}  // namespace simpla

#endif /* LOAD_PARTICLE_H_ */
