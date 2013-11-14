/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 * $Id$
 * Fluid/OhmLaw
 *
 *  Created on: 2010-12-6
 *      Author: salmon
 */

#include "ohm_law.h"
#include <string>

namespace simpla
{
namespace Fluid
{
using namespace fetl;

template<typename TV, typename TG, typename TBUNBLE>
void ColdFluid(Field<IZeroForm, nTuple<THREE, TV>, TG> & Bv,
		Field<IZeroForm, nTuple<THREE, TV>, TG> &Ev,
		Field<IZeroForm, nTuple<THREE, TV>, TG> &Jv, TBUNBLE &ps, Real dt)
{
	DEFINE_FIELDS(TV, TG);

	const Real epsilon0 = 1.0;
	const Real protonMass = 1.0;
	const Real elementaryCharge = 1.0;

	VecZeroForm K_(Ev.grid);
	VecZeroForm dEv_(Ev.grid);

	// vector fields on  grid node

	ZeroForm BB(Bv.grid);
	ZeroForm pa_(Bv.grid);
	ZeroForm pb_(Bv.grid);
	ZeroForm pc_(Bv.grid);
	ZeroForm pa1_(Bv.grid);
	ZeroForm pb1_(Bv.grid);
	ZeroForm pc1_(Bv.grid);

	BB = Dot(Bv, Bv);

	pa1_ = 0.0;
	pb1_ = 0.0;
	pc1_ = 0.0;

	dEv_ = 0.0;

	for (typename TBUNBLE::iterator it = ps.begin(); it != ps.end(); ++it)
	{

		Real m = it->m * protonMass;
		Real Z = it->Z * elementaryCharge;
		Real as = 2.0 * m / (dt * Z);

		pa1_ += it->ns * Z / as;
		pb1_ += it->ns * Z / (BB + as * as);
		pc1_ += it->ns * Z / ((BB + as * as) * as);

	}
	pa1_ *= (0.5 * dt) / epsilon0;
	pb1_ *= (0.5 * dt) / epsilon0;
	pc1_ *= (0.5 * dt) / epsilon0;

	pa1_ += 1.0;

	pa_ = 1.0 / pa1_;

	pb_ = -pb1_ / ((pc1_ * BB - pa1_) * (pc1_ * BB - pa1_) + pb1_ * pb1_ * BB);

	pc_ = -(-pc1_ * pc1_ * BB + pc1_ * pa1_ - pb1_ * pb1_)
			/ (pa1_
					* ((pc1_ * BB - pa1_) * (pc1_ * BB - pa1_)
							+ pb1_ * pb1_ * BB));

	for (typename TBUNBLE::iterator it = ps.begin(); it != ps.end(); ++it)
	{

		Real m = it->m * protonMass;
		Real Z = it->Z * elementaryCharge;
		Real as = 2.0 * m / (dt * Z);

		dEv_ -= it->J1v * 0.5 * dt / epsilon0;

		K_ = it->J1v * as + Cross(it->J1v, Bv) + ((Ev) * it->ns) * Z;

		it->J1v = K_ / as + Cross(K_, Bv) / (BB + as * as)
				+ Cross(Cross(K_, Bv), Bv) / (as * (BB + as * as));

		dEv_ -= it->J1v * 0.5 * dt / epsilon0;
	}

	K_ = Ev + dEv_;

	Ev = K_ * pa_ + Cross(K_, Bv) * pb_ + Cross(Cross(K_, Bv), Bv) * pc_;

	for (typename TBUNBLE::iterator it = ps.begin(); it != ps.end(); ++it)
	{
		Real m = ps.m * protonMass;
		Real Z = ps.Z * elementaryCharge;
		Real as = 2.0 * m / (dt * Z);

		it->J1v += (Ev / as + Cross(Ev, Bv) / (BB + as * as)
				+ Cross(Cross(Ev, Bv), Bv) / (as * (BB + as * as)))
				* (it->ns * Z);
	}

}
} // namespace Fluid
}  // namespace simpla
