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
#include "emfield/boundaryCondition.h"

namespace Fluid
{
OhmLaw::OhmLaw(Context::Holder pctx) :
		ctx(pctx)
{

}
OhmLaw::~OhmLaw()
{
}

void OhmLaw::pre_process(std::list<std::string> const & splist)
{

	for (std::list<std::string>::const_iterator it = splist.begin();
			it != splist.end(); ++it)
	{

		if (ctx->species.find(*it) != ctx->species.end()
				&& boost::any_cast<std::string>(ctx->species[*it]["engine"])
						== "ColdFluid")
		{
			splist_.push_back(*it);
		}

	}

	Context::OneForm::Holder E1 = ctx->getField<IOneForm, Real>("E1");

	ctx->getField<IVecZeroForm, Real>("Ev").swap(Ev);

	*Ev = 0.0;

	if (E1 != Context::OneForm::Holder())
	{
		*Ev = ToVecZeroForm(*E1);
	}
	else
	{
		*Ev = 0.0;
	}
}

void OhmLaw::process()
{
	DINGDONG;
	if (splist_.empty())
	{
		return;
	}

	Context::TwoForm &B1 = *ctx->getField<ITwoForm, Real>("B1");
	Context::OneForm &E1 = *ctx->getField<IOneForm, Real>("E1");

	Context::VecZeroForm Bv(*(ctx->grid));
	Context::VecZeroForm K_(*(ctx->grid));
	// vector fields on  grid node

	Context::ZeroForm BB(*(ctx->grid));
	Context::ZeroForm pa_(*(ctx->grid));
	Context::ZeroForm pb_(*(ctx->grid));
	Context::ZeroForm pc_(*(ctx->grid));
	Context::ZeroForm pa1_(*(ctx->grid));
	Context::ZeroForm pb1_(*(ctx->grid));
	Context::ZeroForm pc1_(*(ctx->grid));

	Scalar dt = ctx->grid->dt;
	Bv = ToVecZeroForm(*ctx->getField<ITwoForm, Real>("B0"));
	BB = Dot(Bv, Bv);

	pa1_ = 0.0;
	pb1_ = 0.0;
	pc1_ = 0.0;

	Context::VecZeroForm dEv_(*(ctx->grid));

	dEv_ = ToVecZeroForm(*ctx->getField<IOneForm, Real>("dE1"));

	for (std::list<std::string>::const_iterator it = splist_.begin();
			it != splist_.end(); ++it)
	{

		Context::ZeroForm & ns = *ctx->getField<IZeroForm, Real>(*it + "_ns");

		Real m = boost::any_cast<Real>(ctx->species[*it]["m"])
				* ctx->ProtonMass;

		Real Z = boost::any_cast<Real>(ctx->species[*it]["Z"])
				* ctx->ElementaryCharge;

		Scalar as = 2.0 * m / (dt * Z);

		pa1_ += ns * Z / as;
		pb1_ += ns * Z / (BB + as * as);
		pc1_ += ns * Z / ((BB + as * as) * as);

	}
	pa1_ *= (0.5 * dt) / ctx->Epsilon0;
	pb1_ *= (0.5 * dt) / ctx->Epsilon0;
	pc1_ *= (0.5 * dt) / ctx->Epsilon0;

	pa1_ += 1.0;

	pa_ = 1.0 / pa1_;

	pb_ = -pb1_ / ((pc1_ * BB - pa1_) * (pc1_ * BB - pa1_) + pb1_ * pb1_ * BB);

	pc_ = -(-pc1_ * pc1_ * BB + pc1_ * pa1_ - pb1_ * pb1_)
			/ (pa1_
					* ((pc1_ * BB - pa1_) * (pc1_ * BB - pa1_)
							+ pb1_ * pb1_ * BB));

	for (std::list<std::string>::const_iterator it = splist_.begin();
			it != splist_.end(); ++it)
	{

		Context::ZeroForm & ns = *ctx->getField<IZeroForm, Real>(*it + "_ns");

		Context::VecZeroForm & J1v = *ctx->getField<IVecZeroForm, Real>(*it + "_J1v");

		Real m = boost::any_cast<Real>(ctx->species[*it]["m"])
				* ctx->ProtonMass;

		Real Z = boost::any_cast<Real>(ctx->species[*it]["Z"])
				* ctx->ElementaryCharge;

		Scalar as = 2.0 * m / (dt * Z);

		dEv_ -= J1v * 0.5 * dt / ctx->Epsilon0;

		K_ = J1v * as + Cross(J1v, Bv) + ((*Ev) * ns) * Z;

		J1v = K_ / as + Cross(K_, Bv) / (BB + as * as)
				+ Cross(Cross(K_, Bv), Bv) / (as * (BB + as * as));

		dEv_ -= J1v * 0.5 * dt / ctx->Epsilon0;
	}

	K_ = *Ev + dEv_;

	*Ev = K_ * pa_ + Cross(K_, Bv) * pb_ + Cross(Cross(K_, Bv), Bv) * pc_;

	for (std::list<std::string>::const_iterator it = splist_.begin();
			it != splist_.end(); ++it)
	{

		Context::ZeroForm & ns = *ctx->getField<IZeroForm, Real>(*it + "_ns");

		Context::VecZeroForm & J1v = *ctx->getField<IVecZeroForm, Real>(*it + "_J1v");

		Real m = boost::any_cast<Real>(ctx->species[*it]["m"])
				* ctx->ProtonMass;

		Real Z = boost::any_cast<Real>(ctx->species[*it]["Z"])
				* ctx->ElementaryCharge;

		Scalar as = 2.0 * m / (dt * Z);

		J1v += (*Ev / as + Cross(*Ev, Bv) / (BB + as * as)
				+ Cross(Cross(*Ev, Bv), Bv) / (as * (BB + as * as))) * (ns * Z);

		ctx->communicateField(*it + "_J1v");
//		*sp.ns -= Diverge((*sp.J1)) * ctx_->grid->dt;

	}

	E1 = ToOneForm(*Ev);

	ctx->communicateField("Ev");
	ctx->communicateField("E1");

}
} // namespace Fluid
