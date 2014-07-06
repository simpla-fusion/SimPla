/*
 * fluid_cold_engine.h
 *
 * \date  2014-4-15
 *      \author  salmon
 */

#ifndef FLUID_COLD_ENGINE_H_
#define FLUID_COLD_ENGINE_H_
#include <functional>

#include "../../src/fetl/fetl.h"
#include "../../src/fetl/load_field.h"
#include "../../src/fetl/save_field.h"

namespace simpla
{

template<typename > class ColdFluid;
template<typename > class Particle;

/**
 * \ingroup ParticleEngine
 * \brief Cold Plasma fluid
 */
template<typename TM>
class Particle<ColdFluid<TM> >
{
public:
	static constexpr int IForm = VERTEX;

	typedef TM mesh_type;

	typedef ColdFluid<mesh_type> engine_type;

	typedef Particle<engine_type> this_type;

	typedef ParticleBase<mesh_type> base_type;

	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type::iterator iterator;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef typename mesh_type:: template field<VERTEX, scalar_type> n_type;

	typedef typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> J_type;

	enum
	{
		EnableImplicit = true
	};
	const Real m;
	const Real q;

	mesh_type const & mesh;

	n_type n;
	J_type J;

	template<typename TDict>
	Particle(TDict const & dict, mesh_type const & pmesh);

	~Particle();

	template<typename TDict, typename TModel, typename ...Args>
	static std::shared_ptr<this_type> Create(TDict dict, TModel const & model, Args && ... args)
	{
		std::shared_ptr<this_type> res(new this_type(dict, model.mesh));

		try
		{
			res->J.clear();
			res->n.clear();

			LoadField(dict["Density"], &(res->n));
			LoadField(dict["Current"], &(res->J));

			res->n *= res->q;
		} catch (...)
		{
			PARSER_ERROR("Configure  Particle<ColdFluid> error!");
		}

		return res;
	}

	static std::string GetTypeAsString()
	{
		return "ColdFluid";
	}

	void NextTimeStepZero(typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & E,
	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & B);

	void NextTimeStepHalf(typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & E,
	        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & B);

	std::string Save(std::string const & name, bool is_verbose = false) const;

private:

}
;

template<typename TM>
template<typename TDict> Particle<ColdFluid<TM>>::Particle(TDict const & dict, mesh_type const & pmesh)
		: mesh(pmesh),

		m(dict["Mass"].template as<Real>(1.0)),

		q(dict["Charge"].template as<Real>(1.0)),

		n(mesh), J(mesh)
{
}

template<typename TM>
Particle<ColdFluid<TM>>::~Particle()
{
}

template<typename TM>
std::string Particle<ColdFluid<TM>>::Save(std::string const & path, bool is_verbose) const
{
	std::stringstream os;

	GLOBAL_DATA_STREAM.OpenGroup(path );

	DEFINE_PHYSICAL_CONST;

	if (is_verbose)
	{
		os

		<< "Engine = '" << GetTypeAsString()

		<< " , " << "Mass = " << m / proton_mass << " * m_p"

		<< " , " << "Charge = " << q / elementary_charge << " * q_e"

		;
	}

	os << "\n, n =" << simpla::Save("n", n);

	os << "\n, J =" << simpla::Save("J", J);

	return os.str();
}
template<typename TM>
void Particle<ColdFluid<TM>>::NextTimeStepZero(
        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & E,
        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & B)
{
	LOGGER << "Push particles Step Zero[ " << GetTypeAsString() << "]";
	Real dt = mesh.GetDt();
	LOG_CMD(n -= Diverge(MapTo<EDGE>(J)) * dt);
}
template<typename TM>
void Particle<ColdFluid<TM>>::NextTimeStepHalf(
        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & E,
        typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> const & B)
{
	LOGGER << "Push particles Step Half[ " << GetTypeAsString() << "]";

	auto K = mesh.template make_field<VERTEX, nTuple<3, scalar_type>>();

	Real as = 0.5 * q / m * mesh.GetDt();

	K = J + Cross(J, B) * as + 2.0 * as * n * E;

	J = (K + Cross(K, B) * as + B * (Dot(K, B) * as * as)) / (Dot(B, B) * as * as + 1);

}

}
// namespace simpla

#endif /* FLUID_COLD_ENGINE_H_ */
