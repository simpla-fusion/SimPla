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
class Particle<ColdFluid<TM>> : public ParticleBase<TM>
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

	mesh_type const & mesh;
	Field<mesh_type, IForm, nTuple<3, scalar_type>> Bv;

	template<typename ...Args>
	Particle(mesh_type const & pmesh, Args const & ...);

	~Particle();

	template<typename TDict, typename ...Others>
	void Load(TDict const & dict, Others const &...);

	static std::string GetTypeAsString()
	{
		return "ColdFluid";
	}

	std::string GetTypeAsString_() const
	{
		return GetTypeAsString();
	}
	inline Real GetMass() const
	{
		return m_;
	}

	Real GetCharge() const
	{
		return q_;
	}

	void NextTimeStep(Field<mesh_type, EDGE, scalar_type> const & E, Field<mesh_type, FACE, scalar_type> const & B);

	std::string Dump(std::string const & name, bool is_verbose) const;

private:
	Real m_;
	Real q_;

	bool enableNonlinear_;
}
;

template<typename TM>
template<typename ...Args> Particle<ColdFluid<TM>>::Particle(mesh_type const & pmesh, Args const & ...args)
		: base_type(pmesh), mesh(pmesh), q_(1.0), m_(1.0), enableNonlinear_(false), Bv(mesh)
{
	base_type::EnableImplicitPushE();
	Load(std::forward<Args const &>(args)...);
}

template<typename TM>
Particle<ColdFluid<TM>>::~Particle()
{
}

template<typename TM>
template<typename TDict, typename ...Others>
void Particle<ColdFluid<TM>>::Load(TDict const &dict, Others const &...)
{
	m_ = dict["Mass"].template as<Real>(1.0);
	q_ = dict["Charge"].template as<Real>(1.0);

	LoadField(dict["Density"], &(base_type::n));

	base_type::n *= q_;

	LoadField(dict["Current"], &(base_type::J));
}

template<typename TM>
std::string Particle<ColdFluid<TM>>::Dump(std::string const & path, bool is_verbose) const
{
	std::stringstream os;
	if (is_verbose)
	{
		DEFINE_PHYSICAL_CONST(mesh.constants());

		os

		<< "Engine = '" << GetTypeAsString()

		<< " , " << "Mass = " << m_ / proton_mass << " * m_p"

		<< " , " << "Charge = " << q_ / elementary_charge << " * q_e"

		;
	}
	os << base_type::Dump(path, is_verbose);

	return os.str();
}

template<typename TM>
void Particle<ColdFluid<TM>>::NextTimeStep(Field<mesh_type, EDGE, scalar_type> const & E,
        Field<mesh_type, FACE, scalar_type> const & B)
{
	LOGGER << "Push particles [ " << GetTypeAsString() << "]";

	Real dt = mesh.GetDt();

	Bv = MapTo<IForm>(B);

	auto & Jv = base_type::Jv;
	auto & rho = base_type::n;

	Real as = 0.5 * GetCharge() * dt / GetMass();

	Jv += Cross(Jv, Bv) * as + 2.0 * as * rho * MapTo<IForm>(E);

	Jv = (Jv + Cross(Jv, Bv) * as + Bv * (Dot(Jv, Bv) * as * as)) / (Dot(Bv, Bv) * as * as + 1);

	rho -= Diverge(MapTo<EDGE>(Jv)) * dt;
}

}
// namespace simpla

#endif /* FLUID_COLD_ENGINE_H_ */
