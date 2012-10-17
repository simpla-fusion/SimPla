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
#include <string>
#include <map>
#include <list>
#include <boost/foreach.hpp>
#include "include/simpla_defs.h"
#include "fetl/fetl.h"
#include "engine/modules.h"
#include "engine/context.h"
namespace simpla
{
namespace em
{

template<typename TV, typename TG>
class ColdFluid: public Module
{
public:

	typedef ColdFluid<TV, TG> ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	DEFINE_FIELDS(TV, TG)

	ColdFluid(Context<TG> & d, const ptree & pt);
	virtual ~ColdFluid()
	{
	}

	virtual void Eval();
private:
	Context<TG> & ctx;
	Grid const & grid;

	const Real dt;
	const Real mu0;
	const Real epsilon0;
	const Real proton_mass;
	const Real elementary_charge;

	struct Sepcies
	{
		Sepcies(Real pm, Real pZ, TR1::shared_ptr<ZeroForm> n,
				TR1::shared_ptr<VecZeroForm> J) :
				m(pm), Z(pZ), ns(n), Js(J)
		{

		}
		const Real m;
		const Real Z;
		TR1::shared_ptr<ZeroForm> ns;
		TR1::shared_ptr<VecZeroForm> Js;
	};

	std::list<TR1::shared_ptr<Sepcies> > sp_list;

	// vector fields on  grid node

	ZeroForm BB;
	ZeroForm pa_;
	ZeroForm pb_;
	ZeroForm pc_;
	ZeroForm pa1_;
	ZeroForm pb1_;
	ZeroForm pc1_;
//input
	TR1::shared_ptr<VecZeroForm> Jv;
//output
	TR1::shared_ptr<VecZeroForm> Ev;
	TR1::shared_ptr<VecZeroForm> Bv;

};

template<typename TV, typename TG>
ColdFluid<TV, TG>::ColdFluid(Context<TG> & d, const ptree & pt) :
		ctx(d),

		grid(ctx.grid),

		dt(ctx.dt),

		mu0(ctx.PHYS_CONSTANTS["permeability_of_free_space"]),

		epsilon0(ctx.PHYS_CONSTANTS["permittivity_of_free_space"]),

		proton_mass(ctx.PHYS_CONSTANTS["proton_mass"]),

		elementary_charge(ctx.PHYS_CONSTANTS["elementary_charge"]),

		Jv(ctx.template GetObject<VecZeroForm>("Jv")),

		Ev(ctx.template GetObject<VecZeroForm>("Ev")),

		Bv(ctx.template GetObject<VecZeroForm>("Bv")),

		BB(grid),

		pa_(grid), pb_(grid), pc_(grid),

		pa1_(grid), pb1_(grid), pc1_(grid)

{

	pa1_ = 0.0;
	pb1_ = 0.0;
	pc1_ = 0.0;

	BOOST_FOREACH(const ptree::value_type &v, pt.get_child("Composition"))
	{
		std::string id = v.second.get<std::string>("<xmlattr>.id");

		sp_list.push_back(TR1::shared_ptr<Sepcies>(new Sepcies(

		v.second.get<Real>("m"),

		v.second.get<Real>("Z"),

		ctx.template GetObject<ZeroForm>(id + "_ns"),

		ctx.template GetObject<VecZeroForm>(id + "_Js")

		)));
	}
	LOG << "Create module ColdFluid";

}

template<typename TV, typename TG>
void ColdFluid<TV, TG>::Eval()
{
	LOG << "Run module ColdFluid";

	BB = Dot(*Bv, *Bv);

	for (typename std::list<TR1::shared_ptr<Sepcies> >::iterator it =
			sp_list.begin(); it != sp_list.end(); ++it)
	{

		Real m = (*it)->m * proton_mass;
		Real Z = (*it)->Z * elementary_charge;
		Real as = 2.0 * m / (dt * Z);

		pa1_ += *(*it)->ns * Z / as;
		pb1_ += *(*it)->ns * Z / (BB + as * as);
		pc1_ += *(*it)->ns * Z / ((BB + as * as) * as);

	}
	pa1_ = pa1_ * (0.5 * dt) / epsilon0;
	pb1_ = pb1_ * (0.5 * dt) / epsilon0;
	pc1_ = pc1_ * (0.5 * dt) / epsilon0;

	pa1_ = pa1_ + 1.0;

	pa_ = 1.0 / pa1_;

	pb_ = -pb1_ / ((pc1_ * BB - pa1_) * (pc1_ * BB - pa1_) + pb1_ * pb1_ * BB);

	pc_ = -(-pc1_ * pc1_ * BB + pc1_ * pa1_ - pb1_ * pb1_)
			/ (pa1_
					* ((pc1_ * BB - pa1_) * (pc1_ * BB - pa1_)
							+ pb1_ * pb1_ * BB));

	VecZeroForm K_(grid);
	VecZeroForm dEv_(grid);

	for (typename std::list<TR1::shared_ptr<Sepcies> >::iterator it =
			sp_list.begin(); it != sp_list.end(); ++it)
	{

		Real m = (*it)->m * proton_mass;
		Real Z = (*it)->Z * elementary_charge;
		Real as = 2.0 * m / (dt * Z);

		dEv_ -= *(*it)->Js * 0.5 * dt / epsilon0;

		K_ = *(*it)->Js * as + Cross(*(*it)->Js, (*Bv))
				+ ((*Ev) * (*(*it)->ns)) * Z;

		*(*it)->Js = K_ / as + Cross(K_, (*Bv)) / (BB + as * as)
//				+ Cross(Cross(K_, (*Bv)), (*Bv))
//				/ (as * (BB + as * as))
				;

		dEv_ -= *(*it)->Js * 0.5 * dt / epsilon0;
	}

	K_ = (*Ev) + dEv_;

	(*Ev) = K_ * pa_ + Cross(K_, (*Bv)) * pb_
			+ Cross(Cross(K_, (*Bv)), (*Bv)) * pc_;

	for (typename std::list<TR1::shared_ptr<Sepcies> >::iterator it =
			sp_list.begin(); it != sp_list.end(); ++it)
	{
		Real m = (*it)->m * proton_mass;
		Real Z = (*it)->Z * elementary_charge;
		Real as = 2.0 * m / (dt * Z);

		*(*it)->Js += ((*Ev) / as + Cross((*Ev), (*Bv)) / (BB + as * as)
				+ Cross(Cross((*Ev), (*Bv)), (*Bv)) / (as * (BB + as * as)))
				* (*(*it)->ns * Z);
	}

}

} // namespace em
} // namespace simpla

#endif  // SRC_FLUID_OHM_LAW_H_
