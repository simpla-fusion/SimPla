/*
 * fluid_cold_engine.h
 *
 *  Created on: 2014年4月15日
 *      Author: salmon
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

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	typedef Field<mesh_type, VERTEX, scalar_type> n_type;

	typedef Field<mesh_type, VERTEX, nTuple<3, scalar_type>> J_type;

	enum
	{
		EnableImplicit = true
	};
	const Real m;
	const Real q;

	mesh_type const & mesh;

	n_type n;
	J_type J;

	template<typename TDict, typename ...Args>
	Particle(mesh_type const & pmesh, TDict const & dict, Args const & ...);

	~Particle();

	static std::string GetTypeAsString()
	{
		return "ColdFluid";
	}

	void NextTimeStepZero(Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & E,
	        Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & B);

	void NextTimeStepHalf(Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & E,
	        Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & B);

	std::string Dump(std::string const & name, bool is_verbose) const;

private:

}
;

template<typename TM>
template<typename TDict, typename ...Args> Particle<ColdFluid<TM>>::Particle(mesh_type const & pmesh,
        TDict const & dict, Args const & ...args)
		: mesh(pmesh),

		m(dict["Mass"].template as<Real>(1.0)),

		q(dict["Charge"].template as<Real>(1.0)),

		n(mesh), J(mesh)
{
	J.Clear();
	n.Clear();

	LoadField(dict["Density"], &(n));

	n *= q;

	LoadField(dict["Current"], &(J));

}

template<typename TM>
Particle<ColdFluid<TM>>::~Particle()
{
}

template<typename TM>
std::string Particle<ColdFluid<TM>>::Dump(std::string const & path, bool is_verbose) const
{
	std::stringstream os;

	GLOBAL_DATA_STREAM.OpenGroup(path );

	if (is_verbose)
	{
		DEFINE_PHYSICAL_CONST(mesh.constants());

		os

		<< "Engine = '" << GetTypeAsString()

		<< " , " << "Mass = " << m / proton_mass << " * m_p"

		<< " , " << "Charge = " << q / elementary_charge << " * q_e"

		;
	}

	os << "\n, n =" << simpla::Dump(n, "n", is_verbose);

	os << "\n, J =" << simpla::Dump(J, "J", is_verbose);

	return os.str();
}
template<typename TM>
void Particle<ColdFluid<TM>>::NextTimeStepZero(Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & E,
        Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & B)
{
	LOGGER << "Push particles Step Zero[ " << GetTypeAsString() << "]";
	Real dt = mesh.GetDt();
	LOG_CMD(n -= Diverge(MapTo<EDGE>(J)) * dt);
}
template<typename TM>
void Particle<ColdFluid<TM>>::NextTimeStepHalf(Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & E,
        Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & B)
{
	LOGGER << "Push particles Step Half[ " << GetTypeAsString() << "]";

	Field<mesh_type, VERTEX, nTuple<3, scalar_type>> K(mesh);

	Real as = 0.5 * q / m * mesh.GetDt();

	K = J + Cross(J, B) * as + 2.0 * as * n * E;

	J = (K + Cross(K, B) * as + B * (Dot(K, B) * as * as)) / (Dot(B, B) * as * as + 1);

}

}
// namespace simpla

#endif /* FLUID_COLD_ENGINE_H_ */
