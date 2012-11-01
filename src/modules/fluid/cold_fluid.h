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
#include "utilities/properties.h"

namespace simpla
{
namespace fliud
{

template<typename TG>
class ColdFluid
{
public:

	typedef ColdFluid<TG> ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	DEFINE_FIELDS(typename TG::ValueType, TG)

	ColdFluid(Context<TG> & d, const ptree & pt);

	virtual ~ColdFluid();

	static TR1::function<void()> Create(Context<TG> * d, const ptree & pt)
	{
		return TR1::bind(&ThisType::Eval,
				TR1::shared_ptr<ThisType>(new ThisType(*d, pt)));
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

	std::map<std::string, std::string> dataflow_;
	// internal variable name, type, reference name

};

template<typename TG>
ColdFluid<TG>::ColdFluid(Context<TG> & d, const ptree & pt) :
		ctx(d),

		grid(ctx.grid),

		dt(ctx.grid.dt),

		mu0(ctx.PHYS_CONSTANTS["permeability_of_free_space"]),

		epsilon0(ctx.PHYS_CONSTANTS["permittivity_of_free_space"]),

		proton_mass(ctx.PHYS_CONSTANTS["proton_mass"]),

		elementary_charge(ctx.PHYS_CONSTANTS["elementary_charge"])

{
	LOG << "Create module ColdFluid";
	BOOST_FOREACH(const typename ptree::value_type &v, pt.get_child("Data"))
	{
		dataflow_[v.second.get<std::string>("<xmlattr>.Name")] =
				v.second.get_value<std::string>();

	}

	BOOST_FOREACH(
			const typename ptree::value_type &v, pt.get_child("Arguments"))
	{
		std::string id = v.second.get<std::string>("<xmlattr>.Name");

		sp_list.push_back(TR1::shared_ptr<Sepcies>(new Sepcies(

		v.second.template get<Real>("m"),

		v.second.template get<Real>("Z"),

		ctx.template GetObject<ZeroForm>(id + "_ns"),

		ctx.template GetObject<VecZeroForm>(id + "_Js")

		)));
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

	TwoForm const&B = *ctx.template GetObject<TwoForm>(dataflow_["B"]);
	OneForm const&E = *ctx.template GetObject<OneForm>(dataflow_["E"]);
	OneForm &J = *ctx.template GetObject<OneForm>(dataflow_["J"]);

	ZeroForm BB(grid);

	VecZeroForm Ev(grid), Bv(grid), dEvdt(grid);

	Bv = MapTo(Int2Type<IZeroForm>(), B);

	Ev = MapTo(Int2Type<IZeroForm>(),
			E + (Curl(B) / mu0 - J) / epsilon0 * (dt * 0.5));

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

	for (typename std::list<TR1::shared_ptr<Sepcies> >::iterator it =
			sp_list.begin(); it != sp_list.end(); ++it)
	{

		Real m = (*it)->m * proton_mass;
		Real Z = (*it)->Z * elementary_charge;
		Real as = 2.0 * m / (dt * Z);

		a += *(*it)->ns * Z / as;
		b += *(*it)->ns * Z / (BB + as * as);
		c += *(*it)->ns * Z / ((BB + as * as) * as);

		K_ = // 2.0*nu**(*it)->Js
				-2.0 * Cross(*(*it)->Js, Bv) - (Ev * (*(*it)->ns)) * (2.0 * Z);

		K -= *(*it)->Js
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

	for (typename std::list<TR1::shared_ptr<Sepcies> >::iterator it =
			sp_list.begin(); it != sp_list.end(); ++it)
	{
		Real ms = (*it)->m * proton_mass;
		Real Zs = (*it)->Z * elementary_charge;
		Real as = 2.0 * ms / (dt * Zs);

		K_ = // 2.0*nu*(*(*it)->Js)
				-2.0 * Cross(*(*it)->Js, Bv)
						- (2.0 * Ev + dEvdt * dt) * (*(*it)->ns) * Zs;
		*(*it)->Js += K_ / as + Cross(K_, Bv) / (BB + as * as)
				+ Cross(Cross(K_, Bv), Bv) / (as * (BB + as * as));
	}

	J -= MapTo(Int2Type<IOneForm>(), dEvdt);

}

} // namespace fliud
} // namespace simpla

#endif  // SRC_FLUID_OHM_LAW_H_
