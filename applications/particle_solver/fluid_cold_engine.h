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

	template<typename ...Args>
	Particle(mesh_type const & pmesh, Args const & ...);

	~Particle();

	template<typename TDict, typename ...Others>
	void Load(TDict const & dict, Others const &...);

	static std::string GetTypeAsString()
	{
		return "ColdFluid";
	}

	//**************************************************************************************************
	// Interface

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

	bool EnableImplicit() const
	{
		return true;
	}
	void Accept(VisitorBase const& visitor)
	{
		CHECK("Accept Visitor");
	}
	void NextTimeStep(
			Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & E,
			Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & B);

	std::string Dump(std::string const & name, bool is_verbose) const;

	Field<mesh_type, VERTEX, scalar_type> & n()
	{
		return n_;
	}
	Field<mesh_type, VERTEX, scalar_type> const& n() const
	{
		return n_;
	}
	Field<mesh_type, EDGE, scalar_type> &J()
	{
		return J_;
	}
	Field<mesh_type, EDGE, scalar_type> const&J() const
	{
		return J_;
	}
	Field<mesh_type, VERTEX, nTuple<3, scalar_type>> &Jv()
	{
		return Jv_;
	}
	Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const&Jv() const
	{
		return Jv_;
	}

private:
	Real m_;
	Real q_;

	bool enableNonlinear_;

	Field<mesh_type, VERTEX, scalar_type> n_;

	Field<mesh_type, EDGE, scalar_type> J_;

	Field<mesh_type, VERTEX, nTuple<3, scalar_type>> Jv_;
}
;

template<typename TM>
template<typename ...Args> Particle<ColdFluid<TM>>::Particle(
		mesh_type const & pmesh, Args const & ...args) :
		mesh(pmesh), q_(1.0), m_(1.0), enableNonlinear_(false),

		n_(mesh), J_(mesh), Jv_(mesh)
{
	Jv_.Clear();
	n_.Clear();
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
	enableNonlinear_ = dict["EnableNonlinear"].template as<bool>(false);
	LoadField(dict["Density"], &(n_));

	n_ *= q_;

	LoadField(dict["Current"], &(J_));
}

template<typename TM>
std::string Particle<ColdFluid<TM>>::Dump(std::string const & path,
		bool is_verbose) const
{
	std::stringstream os;

	GLOBAL_DATA_STREAM.OpenGroup(path );

	if (is_verbose)
	{
		DEFINE_PHYSICAL_CONST(mesh.constants());

		os

		<< "Engine = '" << GetTypeAsString()

		<< " , " << "Mass = " << m_ / proton_mass << " * m_p"

		<< " , " << "Charge = " << q_ / elementary_charge << " * q_e"

		;
	}

	os << "\n, n =" << simpla::Dump(n_, "n", is_verbose);

	os << "\n, Jv =" << simpla::Dump(Jv_, "Jv", is_verbose);

	return os.str();
}

template<typename TM>
void Particle<ColdFluid<TM>>::NextTimeStep(
		Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & E,
		Field<mesh_type, VERTEX, nTuple<3, scalar_type>> const & B)
{
	LOGGER << "Push particles [ " << GetTypeAsString() << "]";

	Real dt = mesh.GetDt();

	Real as = 0.5 * GetCharge() * dt / GetMass();

	Jv_ += Cross(Jv_, B) * as + 2.0 * as * n_ * E;

	Jv_ = (Jv_ + Cross(Jv_, B) * as + B * (Dot(Jv_, B) * as * as))
			/ (Dot(B, B) * as * as + 1);

}

}
// namespace simpla

#endif /* FLUID_COLD_ENGINE_H_ */
