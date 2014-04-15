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
class Particle<ColdFluid<TM>>
{
	static constexpr int IForm = VOLUME;

	typedef ColdFluid<TM> engine_type;

	typedef Particle<engine_type> this_type;

	typedef typename TM mesh_type;

	typedef typename mesh_type::scalar_type scalar_type;

	typedef typename mesh_type::index_type index_type;

	typedef typename mesh_type::coordinates_type coordinates_type;
public:
	mesh_type const &mesh;

	Real m_;
	Real q_;
	Field<mesh_type, VOLUME, scalar_type> n_;
	Field<mesh_type, VOLUME, nTuple<3, scalar_type>> J_;

	typedef Field<mesh_type, VOLUME, nTuple<3, scalar_type>> vector_field_type;

	bool enableNonlinear_;
	template<typename ...Args> Particle(mesh_type const & pmesh, Args const & ...);

	virtual ~Particle();

	template<typename TWrap, typename TDict, typename ...Args> static //
	bool CreateWrap(TWrap* res, TDict const & dict, mesh_type const & mesh, Args const & ...args)
	{
		bool isDone = false;

		if (dict["Type"].template as<std::string>() == engine_type::TypeName())
		{

			typedef typename TWrap::TE TE;
			typedef typename TWrap::TB TB;
			typedef typename TWrap::TN TN;
			typedef typename TWrap::TJ TJ;

			auto particle = std::shared_ptr<this_type>(new this_type(mesh));

			particle->Load(dict, std::forward<Args const &>(args)...);

			using namespace std::placeholders;

			res->NextTimeStep_ = std::bind(&this_type::template NextTimeStep<TN, TJ, TE, TB>, particle, _1, _2, _3, _4,
			        _5);

			res->Print = std::bind(&this_type::Print, particle, _1);

			res->Dump = std::bind(&this_type::Dump, particle, _1, false);

			isDone = true;
		}

		return isDone;
	}
	template<typename ...Args> void Load(Args const &... args);

	void Print(std::ostream & os) const;

	void Dump(std::string const & path) const;

	void Update();

	template<typename TN, typename TJ, typename TE, typename TB, typename ...Args>
	void NextTimeStep(Real dt, TN * n, TJ * J, TE const &E, TB const &B, Args const& ... args);
}
;

template<typename TM>
template<typename ...Args> Particle<ColdFluid<TM>>::Particle(mesh_type const & pmesh, Args const & ...args)
		: mesh(pmesh), m_(1.0), q_(1.0), n_(mesh), J_(mesh), enableNonlinear_(false)
{
}

template<typename TM>
Particle<ColdFluid<TM>>::~Particle()
{
}

template<typename TM>
template<typename ...Args>
void Particle<ColdFluid<TM>>::Load(Args const & ... args)
{
	Update();
}

template<typename TM>
void Particle<ColdFluid<TM>>::Print(std::ostream & os) const
{

	os

	<< " = { " << " Mass =" << m_ << "," << " Charge =" << q_ << ",\n_"

	<< "\t n_ = " << simpla::Dump(n_, false) << "\n_"

	<< "\t J_ = " << simpla::Dump(J_, false) << "\n_"

	<< "\t},\n_";

}
template<typename TM>
void Particle<ColdFluid<TM>>::Dump(std::string const & path) const
{

}
template<typename TM>
std::ostream & operator<<(std::ostream & os, std::pair<std::string, Particle<TM>> const &self)
{
	return self.Save(os);
}

template<typename TM>
void Particle<ColdFluid<TM>>::Update()
{

}

template<typename TM>
template<typename TN, typename TJ, typename TE, typename TB, typename ...Args>
void Particle<ColdFluid<TM>>::NextTimeStep(Real dt, TN * n, TJ * J, TE const &E, TB const &B, Args const& ... args)
{

	vector_field_type K(mesh);

	vector_field_type Ev(mesh);

	vector_field_type Bv(mesh);

	Bv = MapTo<IForm>(B);

	Ev = MapTo<IForm>(E);

	Real as = (dt * q_) / (2.0 * m_);

	*J -= MapTo<TJ>(J_);

	K = Cross(J_, B) * as + Ev * n_ * q_ * as + J_;

	*J -= (K + Cross(K, B) * as + Bv * (Dot(K, Bv) * as * as)) / (Dot(Bv, Bv) * as * as + 1);

	*n += n_;

	J_ += (Ev + Cross(Ev, Bv) * as + Bv * (Dot(Ev, Bv) * as * as)) * (as * n_ * q_) / (Dot(Bv, Bv) * as * as + 1);

}

}  // namespace simpla

#endif /* FLUID_COLD_ENGINE_H_ */
