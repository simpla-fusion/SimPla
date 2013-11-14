/*
 * cold_fluid.h
 *
 *  Created on: 2013年11月13日
 *      Author: salmon
 */

#ifndef COLD_FLUID_H_
#define COLD_FLUID_H_

#include "fetl/fetl.h"

namespace simpla
{

template<typename TM>
class ColdFluidEM
{
	typedef TM mesh_type;

	template<int IFORM> using Form = Field<Geometry<mesh_type,IFORM>,Real >;
	template<int IFORM> using VForm = Field<Geometry<mesh_type,IFORM>,nTuple<3,Real> >;

	mesh_type const & mesh;

	const double mu0;
	const double epsilon0;
	const double speed_of_light;
	const double proton_mass;
	const double elementary_charge;

	ColdFluidEM(mesh_type const & pm, PhysicalConstants const &phys) :
			mesh(pm)

			, mu0(phys["permeability_of_free_space"])

			, epsilon0(phys["permittivity_of_free_space"])

			, speed_of_light(phys["speed_of_light"])

			, proton_mass(phys["proton_mass"])

			, elementary_charge(phys["elementary_charge"])

			, K_(mesh), K(mesh), a(mesh), b(mesh), c(mesh)

			, BB(mesh), Ev(mesh), Bv(mesh), dEvdt(mesh)
	{
	}

	Form<0, nTuple<3, Real> > K_;
	Form<0, nTuple<3, Real> > K;
	Form<0> a;
	Form<0> b;
	Form<0> c;
	Form<0> BB;
	Form<0, Vec3> Ev, Bv, dEvdt;

	template<typename TE, typename TB, typename TJ, typename TS>
	void Eval(TE const &E, TB const &B, TJ const &J, TS & sp_list, Real dt)
	{
		E += (Curl(B / mu0) - J) / epsilon0 * dt;

		B -= Curl(E) * dt;

		a.clear();
		b.clear();
		c.clear();
		K.clear();

		BB = Wedge(B, HodgeStar(B));

		for (auto &v : sp_list)
		{

			auto & ns = v.get<Form<0> >("n");
			auto & Js = v.get<VForm<0> >("J");
			auto ms = v.properties.get<Real>("m") * proton_mass;
			auto Zs = v.properties.get<Real>("Z") * elementary_charge;

			Form<0> as(mesh);

			as = 2.0 * ms / (dt * Zs);

			a += ns * Zs / as;
			b += ns * Zs / (BB + as * as);
			c += ns * Zs / ((BB + as * as) * as);

			K_ = /* 2.0 * nu * Js*/
			-2.0 * Cross(Js, Bv) - (Ev * ns) * (2.0 * Zs);

			K -= Js + 0.5 * (

			K_ / as

			+ Cross(K_, Bv) / (BB + as * as)

			+ Cross(Cross(K_, Bv), Bv) / (as * (BB + as * as))

			);
		}

		a = a * (0.5 * dt) / epsilon0 - 1.0;
		b = b * (0.5 * dt) / epsilon0;
		c = c * (0.5 * dt) / epsilon0;

		K /= epsilon0;

		dEvdt = K / a
				+ Cross(K, Bv) * b / ((c * BB - a) * (c * BB - a) + b * b * BB)
				+ Cross(Cross(K, Bv), Bv) * (-c * c * BB + c * a - b * b)
						/ (a * ((c * BB - a) * (c * BB - a) + b * b * BB));
		for (auto &v : sp_list)
		{
			auto & ns = v.get<Form<0> >("n");
			auto & Js = v.get<VForm<0> >("J");
			Real ms = v.properties.get<Real>("m") * proton_mass;
			Real Zs = v.properties.get<Real>("Z") * elementary_charge;

			Form<0> as(mesh);
			as = 2.0 * ms / (dt * Zs);

			K_ = // 2.0*nu*(Js)
					-2.0 * Cross(Js, Bv) - (2.0 * Ev + dEvdt * dt) * ns * Zs;
			Js +=

			K_ / as

			+ Cross(K_, Bv) / (BB + as * as)

			+ Cross(Cross(K_, Bv), Bv) / (as * (BB + as * as));
		}

		//		J -=  dEvdt;
	}

};
}  // namespace simpla

#endif /* COLD_FLUID_H_ */
