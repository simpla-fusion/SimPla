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

	Field<mesh_type, VERTEX, nTuple<3, scalar_type>> Bv, Ev;

	Field<mesh_type, VERTEX, Real> BB;

	template<typename ...Others>
	ImplicitPushE(mesh_type const & m, Others const &...)
			: mesh(m), Bv(mesh), BB(mesh), Ev(mesh)
	{
	}

	template<typename TP>
	void NextTimeStep(Field<mesh_type, EDGE, scalar_type> const &E, Field<mesh_type, FACE, scalar_type> const &B,
	        TP const & particles, Field<mesh_type, EDGE, scalar_type> *pdE);
};
template<typename TM>
template<typename TP>
void ImplicitPushE<TM>::NextTimeStep(Field<mesh_type, EDGE, scalar_type> const &E,
        Field<mesh_type, FACE, scalar_type> const &B, TP const & particles, Field<mesh_type, EDGE, scalar_type> *pdE)
{

	DEFINE_PHYSICAL_CONST;

	Real dt = mesh.GetDt();
	LOGGER << "Implicit Push E ";

	if (Ev.empty())
		Ev = MapTo<VERTEX>(E);

	Bv = MapTo<VERTEX>(B);
	BB = Dot(Bv, Bv);

	Field<mesh_type, VERTEX, nTuple<3, scalar_type>> Q(mesh);
	Field<mesh_type, VERTEX, nTuple<3, scalar_type>> K(mesh);

	Q = MapTo<VERTEX>(*pdE);

	Field<mesh_type, VERTEX, Real> a(mesh);
	Field<mesh_type, VERTEX, Real> b(mesh);
	Field<mesh_type, VERTEX, Real> c(mesh);
	a.Clear();
	b.Clear();
	c.Clear();

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

			K = (Ev * rhos * 2.0 + Cross(Js, Bv)) * as + Js;

			Q -= 0.5 * dt / epsilon0
			        * ((K + Cross(K, Bv) * as + Bv * (Dot(K, Bv) * as * as)) / (BB * as * as + 1) + Js);

			a += rhos * as / (BB * as * as + 1);
			b += rhos * as * as / (BB * as * as + 1);
			c += rhos * as * as * as / (BB * as * as + 1);
		}
	}

	a *= 0.5 * dt / epsilon0;
	b *= 0.5 * dt / epsilon0;
	c *= 0.5 * dt / epsilon0;
	a += 1;

	Field<mesh_type, VERTEX, nTuple<3, scalar_type>> dEv(mesh);

	dEv = (Q * a - Cross(Q, Bv) * b + Bv * (Dot(Q, Bv) * (b * b - c * a) / (a + c * BB))) / (b * b * BB + a * a);

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
