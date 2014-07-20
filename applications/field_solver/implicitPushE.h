/*
 * implicitPushE.h
 *
 * \date  2014-4-16
 *      \author  salmon
 */

#ifndef IMPLICITPUSHE_H_
#define IMPLICITPUSHE_H_

#include "../../src/fetl/fetl.h"
#include "../../src/utilities/log.h"
#include "../../src/physics/physical_constants.h"

namespace simpla
{

/**
 *   \ingroup FieldSolver
 *
 *   \class ImplicitPushE
 *   \brief implicit electric field pusher
 *   \see \ref FDTD_Plasma
 *
 */
template<typename TM>
class ImplicitPushE
{
public:
	typedef TM mesh_type;
	typedef typename mesh_type::scalar_type scalar_type;

	mesh_type const &mesh;

	template<int iform, typename TV> using field=typename mesh_type::template field<iform, TV>;

	field<VERTEX, nTuple<3, scalar_type>> Ev;

	field<VERTEX, nTuple<3, Real>> B0;

	field<VERTEX, Real> BB;

	typedef field<VERTEX, scalar_type> n_type;

	typedef field<VERTEX, nTuple<3, scalar_type>> J_type;

	template<typename ...Others>
	ImplicitPushE(mesh_type const & m, Others const &...)
			: mesh(m), B0(mesh), BB(mesh), Ev(mesh)
	{
	}

	template<typename TP>
	void next_timestep(field<EDGE, Real> const &E0, field<FACE, Real> const & pB0, field<EDGE, scalar_type> const &E,
	        field<FACE, scalar_type> const &B, TP const & particles, field<EDGE, scalar_type> *pdE);
};

/**
 *
 * @param E
 * @param B
 * @param particles
 * @param pdE
 */
template<typename TM>
template<typename TP>
void ImplicitPushE<TM>::next_timestep(field<EDGE, Real> const &E0, field<FACE, Real> const &pB0,
        field<EDGE, scalar_type> const &E, field<FACE, scalar_type> const &B, TP const & particles,
        field<EDGE, scalar_type> *pdE)
{
	{
		bool flag = false;
		for (auto &p : particles)
		{
			flag |= (p.second->is_implicit());
		}

		if (!flag)
			return;
	}

	DEFINE_PHYSICAL_CONST

	Real dt = mesh.get_dt();

	LOGGER << "Implicit Push E ";

	if (Ev.empty())
		Ev = MapTo<VERTEX>(E);

	if (B0.empty())
	{
		B0 = MapTo<VERTEX>(pB0);
		BB = Dot(B0, B0);
	}

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
		if (p.second->is_implicit())
		{
			p.second->next_timestep_zero(Ev, B0);
			p.second->update_fields();

			auto & rhos = p.second->template n<n_type>();
			auto & Js = p.second->template J<J_type>();

			Real ms = p.second->get_mass();
			Real qs = p.second->get_charge();

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
		if (p.second->is_implicit())
		{
			p.second->next_timestep_half(Ev, B0);
		}
	}
	Ev += dEv * 0.5;

	*pdE = MapTo<EDGE>(dEv);

	LOGGER << DONE;

}

}
// namespace simpla

#endif /* IMPLICITPUSHE_H_ */
