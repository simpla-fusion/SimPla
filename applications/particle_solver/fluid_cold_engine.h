/*
 * fluid_cold_engine.h
 *
 *  Created on: 2014年4月15日
 *      Author: salmon
 */

#ifndef FLUID_COLD_ENGINE_H_
#define FLUID_COLD_ENGINE_H_
#include <functional>

#include "../../../src/fetl/fetl.h"
#include "../../../src/fetl/load_field.h"
#include "../../../src/fetl/save_field.h"

namespace simpla
{

template<typename > class ColdFluid;
template<typename > class Particle;

template<typename TM>
class Particle<ColdFluid<TM>> : public ParticleBase<TM>
{
public:
	static constexpr int IForm = VOLUME;

	typedef TM mesh_type;

	typedef ColdFluid<mesh_type> engine_type;

	typedef Particle<engine_type> this_type;

	typedef ParticleBase<mesh_type> base_type;

	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;

	template<typename ...Args> Particle(mesh_type const & pmesh, Args const & ...);

	~Particle();

	static std::string GetTypeAsString()
	{
		return "ColdFluid";
	}

	template<typename ...Args> void Load(Args const &... args);

	inline Real GetMass() const
	{
		return m_;
	}

	Real GetCharge() const
	{
		return q_;
	}

	void NextTimeStep(Real dt, Field<mesh_type, EDGE, scalar_type> const E,
	        Field<mesh_type, FACE, scalar_type> const & B);

	void Print(std::ostream & os) const;

	std::string Dump(std::string const & name, bool compact_storage) const;

private:
	Real m_;
	Real q_;

	bool enableNonlinear_;
}
;

template<typename TM>
template<typename ...Args> Particle<ColdFluid<TM>>::Particle(mesh_type const & pmesh, Args const & ...args)
		: base_type(pmesh), q_(1.0), m_(1.0), enableNonlinear_(false)
{
	Load(std::forward<Args const &>(args)...);
}

template<typename TM>
Particle<ColdFluid<TM>>::~Particle()
{
}

template<typename TM>
template<typename ...Args>
void Particle<ColdFluid<TM>>::Load(Args const & ... args)
{
}

template<typename TM>
void Particle<ColdFluid<TM>>::Print(std::ostream & os) const
{

//	os
//
//	<< " = { " << " Mass =" << m_ << "," << " Charge =" << q_ << ",\n_"
//
//	<< "\t n_ = " << simpla::Dump(n, false) << "\n_"
//
//	<< "\t J_ = " << simpla::Dump(J, false) << "\n_"
//
//	<< "\t},\n_";

}

template<typename TM>
std::string Particle<ColdFluid<TM>>::Dump(std::string const & path, bool compact_storage) const
{
	return "";
}

template<typename TM>
void Particle<ColdFluid<TM>>::NextTimeStep(Real dt, Field<mesh_type, EDGE, scalar_type> const E,
        Field<mesh_type, FACE, scalar_type> const & B)
{

//	vector_field_type K(mesh);
//
//	vector_field_type Ev(mesh);
//
//	vector_field_type Bv(mesh);
//
//	Bv = MapTo<IForm>(B);
//
//	Ev = MapTo<IForm>(E);
//
//	Real as = (dt * q_) / (2.0 * m_);
//
//	*J -= MapTo<TJ>(J_);
//
//	K = Cross(J_, B) * as + Ev * n_ * q_ * as + J_;
//
//	*J -= (K + Cross(K, B) * as + Bv * (Dot(K, Bv) * as * as)) / (Dot(Bv, Bv) * as * as + 1);
//
//	*n += n_;
//
//	J_ += (Ev + Cross(Ev, Bv) * as + Bv * (Dot(Ev, Bv) * as * as)) * (as * n_ * q_) / (Dot(Bv, Bv) * as * as + 1);

}

}  // namespace simpla

#endif /* FLUID_COLD_ENGINE_H_ */
