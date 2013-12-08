/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id$
 * MHDOhmLaw.h
 *
 *  Created on: 2010-12-7
 *      Author: salmon
 */

#ifndef SRC_FLUID_OHM_LAW_H_
#define SRC_FLUID_OHM_LAW_H_

#include <memory>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <utility>

#include "../src/simpla_defs.h"
#include "../src/fetl/fetl.h"
#include "../src/utilities/log.h"

namespace simpla
{

template<typename TG>
class ColdFluid
{
public:

	typedef ColdFluid<TG> ThisType;

	typedef std::shared_ptr<ThisType> Holder;

	DEFINE_FIELDS (TG)

	template<typename TConfig>
	ColdFluid(Mesh const &mesh, TConfig const &config);

	~ColdFluid();

	template<typename TConfig> static inline std::function<
			void(Real, Form<1> const &, Form<1>*, Form<2>*)> //
	Create(Mesh const &mesh, const TConfig & pt)
	{
		using namespace std::placeholders;
		return std::bind(&ThisType::Eval,
				std::shared_ptr<ThisType>(new ThisType(mesh, pt)), _1, _2, _3,
				_4);
	}

	void Eval(Real dt, Form<1> const & J, Form<1>*E, Form<2>*B);

	struct Species
	{
		Real m;
		Real Z;
		Form<0> n;
		VectorForm<0> J;
	};

	std::map<std::string, Species> sp_list_;
private:
	Mesh const & mesh_;

};

template<typename TG>
template<typename TConfig>
ColdFluid<TG>::ColdFluid(Mesh const &m, const TConfig & pt) :
		mesh_(m)

{

}
template<typename TG>
ColdFluid<TG>::~ColdFluid()
{
}

template<typename TG>
void ColdFluid<TG>::Eval(Real dt, Form<1> const & J, Form<1>*E, Form<2>*B)
{
	LOG << "Run module ColdFluid";

	const double mu0 = mesh_.phys_constants["permeability of free space"];
	const double epsilon0 = mesh_.phys_constants["permittivity of free space"];
	const double speed_of_light = mesh_.phys_constants["speed of light"];
	const double proton_mass = mesh_.phys_constants["proton mass"];
	const double elementary_charge = mesh_.phys_constants["elementary charge"];

	Form<0> BB(mesh_);

	VectorForm<0> Ev(mesh_), Bv(mesh_), dEvdt(mesh_);

	BB = Dot(Bv, Bv);

	Ev = MapTo<0>(*E);
	Bv = MapTo<0>(*B);

	VectorForm<0> K_(mesh_);
	VectorForm<0> K(mesh_);
	K = 0.0;

	Form<0> a(mesh_);
	Form<0> b(mesh_);
	Form<0> c(mesh_);
	a = 0.0;
	b = 0.0;
	c = 0.0;

	for (auto &v : sp_list_)
	{
		Form<0> & ns = v.second.n;
		VectorForm<0> & Js = v.second.J;

		Real ms = v.second.m * proton_mass;
		Real Zs = v.second.Z * elementary_charge;

		Real as = 2.0 * ms / (dt * Zs);

		a += ns * Zs / as;
		b += ns * Zs / (BB + as * as);
		c += ns * Zs / ((BB + as * as) * as);

		K_ = -2.0 * Cross(Js, Bv) - (Ev * ns) * (2.0 * Zs);

		K -= Js
				+ 0.5
						* (K_ / as + Cross(K_, Bv) / (BB + as * as)
								+ Cross(Cross(K_, Bv), Bv)
										/ (as * (BB + as * as)));

	}

	a = a * (0.5 * dt) / epsilon0 - 1.0;
	b = b * (0.5 * dt) / epsilon0;
	c = c * (0.5 * dt) / epsilon0;

	K /= epsilon0;

	dEvdt = K / a
			+ Cross(K, Bv) * b / ((c * BB - a) * (c * BB - a) + b * b * BB)
			+ Cross(Cross(K, Bv), Bv) * (-c * c * BB + c * a - b * b)
					/ (a * ((c * BB - a) * (c * BB - a) + b * b * BB));
	for (auto &v : sp_list_)

	{
		Form<0> & ns = v.second.n;
		VectorForm<0> & Js = v.second.J;

		Real ms = v.second.m * proton_mass;
		Real Zs = v.second.Z * elementary_charge;

		Real as = 2.0 * ms / (dt * Zs);

		K_ = -2.0 * Cross(Js, Bv) - (2.0 * Ev + dEvdt * dt) * ns * Zs;

		Js += K_ / as + Cross(K_, Bv) / (BB + as * as)
				+ Cross(Cross(K_, Bv), Bv) / (as * (BB + as * as));
	}

}

}
} // namespace simpla

#endif  // SRC_FLUID_OHM_LAW_H_
