/*
 * cold_fluid.h
 *
 *  Created on: 2013年11月13日
 *      Author: salmon
 */

#ifndef COLD_FLUID_H_
#define COLD_FLUID_H_

#include "../../src/fetl/fetl.h"

namespace simpla
{

template<typename TM>
class ColdFluidEM
{
public:
	typedef TM Mesh;
	template<int IFORM> using Form = Field<Geometry<Mesh,IFORM>,Real >;
	template<int IFORM> using VectorForm = Field<Geometry<Mesh,IFORM>,nTuple<3,Real> >;
	template<int IFORM> using TensorForm = Field<Geometry<Mesh,IFORM>,nTuple<3,nTuple<3,Real> > >;
	template<int IFORM> using CForm = Field<Geometry<Mesh,IFORM>,Complex >;
	template<int IFORM> using CVectorForm = Field<Geometry<Mesh,IFORM>,nTuple<3,Complex> >;
	template<int IFORM> using CTensorForm = Field<Geometry<Mesh,IFORM>,nTuple<3,nTuple<3,Complex> > >;

	typedef Mesh mesh_type;

private:

	struct Species
	{
		std::string name;
		Real m;
		Real Z;
		Form<0> n;
		VectorForm<0> J;
	};
	std::list<Species> sp_list_;
public:

	mesh_type const & mesh;

	template<typename TConfig>
	ColdFluidEM(mesh_type const & pmesh, TConfig const &phys) :
			mesh(pmesh)

	{
	}

	template<typename TJ, typename TE, typename TB> inline
	void Eval(Real dt, TJ const &J, TE const *E, TB const *B)
	{

		const double mu0 = mesh.constants["permeability of free space"];
		const double epsilon0 = mesh.constants["permittivity of free space"];
		const double speed_of_light = mesh.constants["speed of light"];
		const double proton_mass = mesh.constants["proton mass"];
		const double elementary_charge = mesh.constants["elementary charge"];

		*E += (Curl((*B) / mu0) - J) / epsilon0 * dt;
		*B -= Curl(*E) * dt;

		VectorForm<0> K_(mesh);
		VectorForm<0> K(mesh);
		Form<0> a(mesh);
		Form<0> b(mesh);
		Form<0> c(mesh);
		Form<0> BB(mesh);
		VectorForm<0> Ev(mesh), Bv(mesh), dEvdt(mesh);

		BB = Wedge(B, HodgeStar(B));

		for (auto &v : sp_list_)
		{

			auto & ns = v.n;
			auto & Js = v.J;
			auto ms = v.m * proton_mass;
			auto Zs = v.Z * elementary_charge;

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

		for (auto &v : sp_list_)
		{
			auto & ns = v.n;
			auto & Js = v.J;
			auto ms = v.m * proton_mass;
			auto Zs = v.Z * elementary_charge;

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
