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

	Field<mesh_type, VERTEX, nTuple<3, scalar_type>> Bv;

	Field<mesh_type, VERTEX, Real> BB;

	template<typename ...Others>
	ImplicitPushE(mesh_type const & m, Others const &...)
			: mesh(m), Bv(mesh), BB(mesh)
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

	DEFINE_PHYSICAL_CONST(mesh.constants());

	Real dt = mesh.GetDt();
	LOGGER << "Implicit Push E ";

	Field<mesh_type, VERTEX, Real> a(mesh);
	Field<mesh_type, VERTEX, Real> b(mesh);
	Field<mesh_type, VERTEX, Real> c(mesh);

	Field<mesh_type, VERTEX, nTuple<3, Real>> Q(mesh);
	Field<mesh_type, VERTEX, nTuple<3, Real>> K(mesh);

//	if (Bv.empty())
	{
		Bv = MapTo<VERTEX>(B);
		BB = Dot(Bv, Bv);
	}
	Q.Clear();
	K.Clear();

	Field<mesh_type, VERTEX, nTuple<3, scalar_type>> Ev(mesh), dEv(mesh);

	Ev = MapTo<VERTEX>(E);
	dEv = MapTo<VERTEX>(*pdE);

	a.Clear();
	b.Clear();
	c.Clear();

	for (auto &p : particles)
	{
		if (!p.second->EnableImplicit())
			continue;

		p.second->NextTimeStepZero(Ev, Bv);

		auto & rhos = p.second->n();
		auto & Js = p.second->Jv();

		Real ms = p.second->GetMass();
		Real qs = p.second->GetCharge();

		Real as = (dt * qs) / (2.0 * ms);

		K = (Ev * rhos * (as * 0.5) + Js);

		Q += (K + Cross(K, Bv) * as + Bv * (Dot(K, Bv) * as * as)) / (BB * as * as + 1);

		a += rhos * as / (BB * as * as + 1);
		b += rhos * as * as / (BB * as * as + 1);
		c += rhos * as * as * as / (BB * as * as + 1);

	}

	a *= 0.5 * dt / epsilon0;
	b *= 0.5 * dt / epsilon0;
	c *= 0.5 * dt / epsilon0;
	a += 1;

	K = Ev + dEv - Q * (dt / epsilon0);

	dEv = (K * a - Cross(K, Bv) * b + Bv * (Dot(K, Bv) * (b * b - c * a) / (a + c * BB))) / (b * b * BB + a * a) - Ev;

	Ev += dEv * 0.5;

	for (auto &p : particles)
	{
		if (p.second->EnableImplicit())
			p.second->NextTimeStepHalf(Ev, Bv);
	}

	*pdE = MapTo<EDGE>(dEv);

	LOGGER << DONE;

}

}
// namespace simpla

#endif /* IMPLICITPUSHE_H_ */
