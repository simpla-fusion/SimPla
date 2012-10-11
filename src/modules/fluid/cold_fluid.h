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

namespace simpla
{
namespace em
{

template<typename TV, typename TG>
class ColdFluid: public Modules
{
public:

	typedef ColdFluid<TV, TG> ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	DEFINE_FIELDS(TV, TG)

	ColdFluid(Domain & d, const ptree & pt);
	virtual ~ColdFluid()
	{
	}

	virtual void Eval();
private:

	Grid const & grid;

	const Real dt;
	const Real mu0;
	const Real epsilon0;
	const Real proton_mass;
	const Real elementary_charge;

	struct Sepcies
	{
		Sepcies(Real pm, Real pZ, ZeroForm & n, VecZeroForm &J) :
				m(pm), Z(pZ), ns(n), Js(J)
		{

		}
		const Real m;
		const Real Z;
		ZeroForm & ns;
		VecZeroForm & Js;
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
	VecZeroForm &Jv;
//output
	VecZeroForm &Ev;
	VecZeroForm &Bv;

};

template<typename TV, typename TG>
ColdFluid<TV, TG>::ColdFluid(Domain & d, const ptree & pt) :
		Modules(d),

		grid(d.grid<UniformRectGrid>()),

		dt(d.dt),

		mu0(d.PHYS_CONSTANTS.get<Real>("mu")),

		epsilon0(d.PHYS_CONSTANTS.get<Real>("epsilon")),

		proton_mass(d.PHYS_CONSTANTS.get<Real>("proton_mass")),

		elementary_charge(d.PHYS_CONSTANTS.get<Real>("elementary_charge")),

		Jv(d.GetObject<VecZeroForm>("")),

		Ev(d.GetObject<VecZeroForm>("")),

		Bv(d.GetObject<VecZeroForm>("")),

		BB(grid),

		pa_(grid), pb_(grid), pc_(grid),

		pa1_(grid), pb1_(grid), pc1_(grid)

{
	using namespace fetl;

	pa1_ = 0.0;
	pb1_ = 0.0;
	pc1_ = 0.0;


	ptree sp_pt = pt.get_child("Species");

//	for (typename ptree::const_iterator it = sp_pt.begin(); it != sp_pt.end();
//			++it)
//	{
//		sp_list.push_back(TR1::shared_ptr<Sepcies>(new Sepcies(
//
//		it.second.get<Real>("m"),
//
//		it.second.get<Real>("Z"),
//
//		d.GetObject<VecZeroForm>(it.first + "ns"),
//
//		d.GetObject<VecZeroForm>(it.first + "Js")
//
//		))
//
//		);
//	}

	for (typename std::list<TR1::shared_ptr<Sepcies> >::iterator it =
			sp_list.begin(); it != sp_list.end(); ++it)
	{

		Real m = (*it)->m * proton_mass;
		Real Z = (*it)->Z * elementary_charge;
		Real as = 2.0 * m / (dt * Z);

		pa1_ += (*it)->ns * Z / as;
		pb1_ += (*it)->ns * Z / (BB + as * as);
		pc1_ += (*it)->ns * Z / ((BB + as * as) * as);

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

}

template<typename TV, typename TG>
void ColdFluid<TV, TG>::Eval()
{

	BB = Dot(Bv, Bv);

	VecZeroForm &K_ = domain.GetObject<VecZeroForm>("");
	VecZeroForm &dEv_ = domain.GetObject<VecZeroForm>("");
	for (typename std::list<TR1::shared_ptr<Sepcies> >::iterator it =
			sp_list.begin(); it != sp_list.end(); ++it)
	{

		Real m = (*it)->m * proton_mass;
		Real Z = (*it)->Z * elementary_charge;
		Real as = 2.0 * m / (dt * Z);

		dEv_ -= (*it)->Js * 0.5 * dt / epsilon0;

		K_ = (*it)->Js * as + Cross((*it)->Js, Bv) + ((Ev) * (*it)->ns) * Z;

		(*it)->Js =
								K_ / as  +
				Cross(K_, Bv) / (BB + as * as)
//				+ Cross(Cross(K_, Bv), Bv)
//				/ (as * (BB + as * as))
				;

		dEv_ -= (*it)->Js * 0.5 * dt / epsilon0;
	}

	K_ = Ev + dEv_;

	Ev = K_ * pa_ + Cross(K_, Bv) * pb_ + Cross(Cross(K_, Bv), Bv) * pc_;

	for (typename std::list<TR1::shared_ptr<Sepcies> >::iterator it =
			sp_list.begin(); it != sp_list.end(); ++it)
	{
		Real m = (*it)->m * proton_mass;
		Real Z = (*it)->Z * elementary_charge;
		Real as = 2.0 * m / (dt * Z);

		(*it)->Js += (Ev / as + Cross(Ev, Bv) / (BB + as * as)
				+ Cross(Cross(Ev, Bv), Bv) / (as * (BB + as * as)))
				* ((*it)->ns * Z);
	}

}

} // namespace em
} // namespace simpla

#endif  // SRC_FLUID_OHM_LAW_H_
