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
#include "engine/context.h"
#include "engine/basemodule.h"
#include "engine/compound.h"

#include "utilities/properties.h"

namespace simpla
{
namespace fliud
{

template<typename TG>
class ColdFluid: public BaseModule
{
public:

	typedef ColdFluid<TG> ThisType;
	typedef std::shared_ptr<ThisType> Holder;

	DEFINE_FIELDS(TG)

	ColdFluid(Context<TG> * d, const PTree & pt);

	virtual ~ColdFluid();

	static std::function<void()> Create(Context<TG> * d, const PTree & pt)
	{
		return std::bind(&ThisType::Eval,
				std::shared_ptr<ThisType>(new ThisType(d, pt)));
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

	std::list<std::shared_ptr<CompoundObject> > sp_list;

	// vector fields on  grid node
};

template<typename TG>
ColdFluid<TG>::ColdFluid(Context<TG> * d, const PTree & pt) :
		BaseModule(d, pt),

		ctx(*d),

		grid(ctx.grid),

		dt(ctx.grid.dt),

		mu0(ctx.PHYS_CONSTANTS["permeability_of_free_space"]),

		epsilon0(ctx.PHYS_CONSTANTS["permittivity_of_free_space"]),

		proton_mass(ctx.PHYS_CONSTANTS["proton_mass"]),

		elementary_charge(ctx.PHYS_CONSTANTS["elementary_charge"])

{
	LOG << "Create module ColdFluid";

	BOOST_FOREACH( const typename PTree::value_type &v, pt){
	if (v.first == "Compound")
	{
		boost::optional<std::shared_ptr<Object> > obj =
		ctx.objects->FindObject(v.second.get_value<std::string>());

		if (!!obj)
		{
			sp_list.push_back(
					std::dynamic_pointer_cast<CompoundObject>(*obj));
		}
	}
}
}
template<typename TG>
ColdFluid<TG>::~ColdFluid()
{
}

template<typename TG>
void ColdFluid<TG>::Eval()
{
	LOG << "Run module ColdFluid";

	TwoForm const&B = *dataset_.get<TwoForm>("B");
	OneForm const&E = *std::dynamic_pointer_cast<OneForm>(dataset_["E"]);
	OneForm &J = *std::dynamic_pointer_cast<OneForm>(dataset_["J"]);

	ZeroForm BB(grid);

	VecZeroForm Ev(grid), Bv(grid), dEvdt(grid);

//	Bv = MapTo(Int2Type<IZeroForm>(), B);
//
//	Ev = MapTo(Int2Type<IZeroForm>(),
//			E + (Curl(B) / mu0 - J) / epsilon0 * (dt * 0.5));

	BB = Dot(Bv, Bv);

	VecZeroForm K_(grid);
	VecZeroForm K(grid);
	K = 0.0;

	ZeroForm a(grid);
	ZeroForm b(grid);
	ZeroForm c(grid);
	a = 0.0;
	b = 0.0;
	c = 0.0;

	for (auto &v : sp_list)
	{
		ZeroForm & ns = v->get<ZeroForm>("n");
		VecZeroForm & Js = v->get<VecZeroForm>("J");

		Real ms = v->properties.get<Real>("m") * proton_mass;
		Real Zs = v->properties.get<Real>("Z") * elementary_charge;

		Real as = 2.0 * ms / (dt * Zs);

		a += ns * Zs / as;
		b += ns * Zs / (BB + as * as);
		c += ns * Zs / ((BB + as * as) * as);

		K_ = // 2.0*nu*Js
				-2.0 * Cross(Js, Bv) - (Ev * ns) * (2.0 * Zs);

		K -= Js
				+ 0.5
						* (K_ / as + Cross(K_, Bv) / (BB + as * as)
								+ Cross(Cross(K_, Bv), Bv)
										/ (as * (BB + as * as)));

	}

	a = a * (0.5 * dt) / epsilon0 - 1.0;
	b = b * (0.5 * dt) / epsilon0;
	c = c * (0.5 * dt) / epsilon0;

	K /= epsilon0;

	dEvdt = K / a
			+ Cross(K, Bv) * b / ((c * BB - a) * (c * BB - a) + b * b * BB)
			+ Cross(Cross(K, Bv), Bv) * (-c * c * BB + c * a - b * b)
					/ (a * ((c * BB - a) * (c * BB - a) + b * b * BB));
	for (auto &v : sp_list)

	{
		ZeroForm & ns = v->get<ZeroForm>("n");
		VecZeroForm & Js = v->get<VecZeroForm>("J");

		Real ms = v->properties.get<Real>("m") * proton_mass;
		Real Zs = v->properties.get<Real>("Z") * elementary_charge;

		Real as = 2.0 * ms / (dt * Zs);

		K_ = // 2.0*nu*(Js)
				-2.0 * Cross(Js, Bv) - (2.0 * Ev + dEvdt * dt) * ns * Zs;
		Js += K_ / as + Cross(K_, Bv) / (BB + as * as)
				+ Cross(Cross(K_, Bv), Bv) / (as * (BB + as * as));
	}

//	J -= MapTo(Int2Type<IOneForm>(), dEvdt);

}

}
// namespace fliud
}// namespace simpla

#endif  // SRC_FLUID_OHM_LAW_H_
