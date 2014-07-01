/*
 * implicitPushE.h
 *
 *  Created on: 2014年4月16日
 *      Author: salmon
 */

#ifndef IMPLICITPUSHE_H_
#define IMPLICITPUSHE_H_

#include "../../src/fetl/fetl.h"
#include "../../src/utilities/log.h"
#include "../../src/physics/physical_constants.h"

namespace simpla
{
template<typename TM>
class ImplicitPushE
{
public:
	typedef TM mesh_type;
	typedef typename mesh_type::scalar_type scalar_type;

	mesh_type const &mesh;

	typename mesh_type:: template field<VERTEX, nTuple<3, scalar_type>> Ev, Bv;

	typename mesh_type:: template field<VERTEX, nTuple<3, Real>> B0;

	typename mesh_type:: template field<VERTEX, Real> BB;

	template<typename ...Others>
	ImplicitPushE(mesh_type const & m, Others const &...)
			: mesh(m), B0(mesh), Bv(mesh), BB(mesh), Ev(mesh)
	{
	}

	template<typename TP>
	void NextTimeStep(typename mesh_type:: template field<EDGE, scalar_type> const &E,
	        typename mesh_type:: template field<FACE, scalar_type> const &B, TP const & particles,
	        typename mesh_type:: template field<EDGE, scalar_type> *pdE);
};
template<typename TM>
template<typename TP>
void ImplicitPushE<TM>::NextTimeStep(typename mesh_type:: template field<EDGE, scalar_type> const &E,
        typename mesh_type:: template field<FACE, scalar_type> const &B, TP const & particles,
        typename mesh_type:: template field<EDGE, scalar_type> *pdE)
{

	DEFINE_PHYSICAL_CONST

	Real dt = mesh.GetDt();

	LOGGER << "Implicit Push E ";

	if (Ev.empty())
		Ev = MapTo<VERTEX>(E);
	Bv = MapTo<VERTEX>(B);
	B0 = real(Bv);
	BB = Dot(B0, B0);

	auto Q = mesh.template make_field<VERTEX, nTuple<3, scalar_type>>();
	auto K = mesh.template make_field<VERTEX, nTuple<3, scalar_type>>();

	Q = MapTo<VERTEX>(*pdE);

	auto a = mesh.template make_field<VERTEX, scalar_type>();
	auto b = mesh.template make_field<VERTEX, scalar_type>();
	auto c = mesh.template make_field<VERTEX, scalar_type>();

	a.clear();
	b.clear();
	c.clear();

	for (auto &p : particles)
	{
		if (p.second->EnableImplicit())
		{
			p.second->NextTimeStepZero(Ev, Bv);

			auto & rhos = p.second->n();
			auto & Js = p.second->Jv();

			Real ms = p.second->GetMass();
			Real qs = p.second->GetCharge();

			Real as = (dt * qs) / (2.0 * ms);

			K = (Ev * rhos * 2.0 + Cross(Js, B0)) * as + Js;

			Q -= 0.5 * dt / epsilon0
			        * ((K + Cross(K, B0) * as + B0 * (Dot(K, B0) * as * as)) / (BB * as * as + 1) + Js);

			a += rhos * as / (BB * as * as + 1);
			b += rhos * as * as / (BB * as * as + 1);
			c += rhos * as * as * as / (BB * as * as + 1);
		}
	}

	a *= 0.5 * dt / epsilon0;
	b *= 0.5 * dt / epsilon0;
	c *= 0.5 * dt / epsilon0;
	a += 1;

	auto dEv = mesh.template make_field<VERTEX, nTuple<3, scalar_type>>();

	dEv = (Q * a - Cross(Q, B0) * b + B0 * (Dot(Q, B0) * (b * b - c * a) / (a + c * BB))) / (b * b * BB + a * a);

	Ev += dEv * 0.5;

	for (auto &p : particles)
	{
		if (p.second->EnableImplicit())
			p.second->NextTimeStepHalf(Ev, Bv);
	}
	Ev += dEv * 0.5;

	*pdE = MapTo<EDGE>(dEv);

	LOGGER << DONE;

}

}
// namespace simpla

#endif /* IMPLICITPUSHE_H_ */
