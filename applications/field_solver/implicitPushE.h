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

template<typename TE, typename TB, typename TP>
void ImplicitPushE(Real dt, TE const &E, TB const &B, TP const & particles, TE *pdE)
{
	typedef typename TE::mesh_type mesh_type;

	mesh_type const & mesh = E.mesh;

	DEFINE_PHYSICAL_CONST(mesh.constants());

	LOGGER << "Implicit Push E:   [ Species Number=" << particles.size() << "]";

	TE & dE = *pdE;

	Field<mesh_type, VERTEX, nTuple<3, Real>> B0(mesh), Ev(mesh);
	Field<mesh_type, VERTEX, Real> BB(mesh);
	Field<mesh_type, VERTEX, Real> a(mesh);
	Field<mesh_type, VERTEX, Real> b(mesh);
	Field<mesh_type, VERTEX, Real> c(mesh);

	Field<mesh_type, VERTEX, nTuple<3, Real>> Q(mesh);
	Field<mesh_type, VERTEX, nTuple<3, Real>> K(mesh);

	B0 = MapTo<VERTEX>(B);
	BB = Dot(B0, B0);
	Q.Clear();
	K.Clear();

	Ev = MapTo<VERTEX>(E);

	a.Clear();
	b.Clear();
	c.Clear();

	for (auto &p : particles)
	{
		if (!p.second->NeedImplicitPushE())
			continue;
		auto & rhos = p.second->n;
		auto & Js = p.second->Jv;

		Real ms = p.second->GetMass();
		Real qs = p.second->GetCharge();

		Real as = (dt * qs) / (2.0 * ms);

		K = (Ev * rhos * (as * 0.5) + Js);

		Q += (K + Cross(K, B0) * as + B0 * (Dot(K, B0) * as * as)) / (BB * as * as + 1);

		a += rhos * as / (BB * as * as + 1);
		b += rhos * as * as / (BB * as * as + 1);
		c += rhos * as * as * as / (BB * as * as + 1);
	}

	a *= 0.5 * dt / epsilon0;
	b *= 0.5 * dt / epsilon0;
	c *= 0.5 * dt / epsilon0;
	a += 1;

	Q = MapTo<VERTEX>(E + dE) - Q * (dt / epsilon0);

	Ev = (Q * a - Cross(Q, B0) * b + B0 * (Dot(Q, B0) * (b * b - c * a) / (a + c * BB))) / (b * b * BB + a * a);

	dE = (MapTo<TE::IForm>(Ev) - E);

	LOGGER << DONE;

}

}  // namespace simpla

#endif /* IMPLICITPUSHE_H_ */
