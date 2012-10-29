/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * Maxwell/Maxwell.h
 *
 *  Created on: 2010-11-16
 *      Author: salmon
 */

#ifndef SRC_EMFIELD_MAXWELL_H_
#define SRC_EMFIELD_MAXWELL_H_

#include "fetl/fetl.h"
#include "fetl/vector_calculus.h"
#include "engine/basecontext.h"
#include "engine/modules.h"
#include "utilities/properties.h"
namespace simpla
{
namespace em
{

template<typename TG>
class Maxwell: public Module
{
public:

	typedef Maxwell<TG> ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	DEFINE_FIELDS(typename TG::ValueType, TG)

	BaseContext & ctx;

	Maxwell(BaseContext & d, ptree const &pt) :
			ctx(d),

			dt(ctx.dt),

			mu0(ctx.PHYS_CONSTANTS["permeability_of_free_space"]),

			epsilon0(ctx.PHYS_CONSTANTS["permittivity_of_free_space"]),

			speed_of_light(ctx.PHYS_CONSTANTS["speed_of_light"]),

			B(pt.get("Parameters.B", "B1")),

			Btype(pt.get("Parameters.B.<xmlattr>.type", "TwoForm")),

			E(pt.get("Parameters.E", "E1")),

			Etype(pt.get("Parameters.E.<xmlattr>.type", "OneForm")),

			J(pt.get("Parameters.J", "J1")),

			Jtype(pt.get("Parameters.J.<xmlattr>.type", "OneForm"))

	{
		LOG << "Create module Maxwell";
	}

	virtual ~Maxwell()
	{
	}

	virtual void Eval()
	{
		LOG << "Run module Maxwell";

		if (Btype == "TwoForm" && Etype == "OneForm" && Jtype == "OneForm")
		{
			DoMaxwellEq(*ctx.template GetObject<TwoForm>(B),
					*ctx.template GetObject<OneForm>(E),
					*ctx.template GetObject<OneForm>(J));
		}
		else if ((Btype == "CTwoForm" || Etype == "COneForm")
				&& Jtype == "OneForm")
		{
			DoMaxwellEq(*ctx.template GetObject<CTwoForm>(B),
					*ctx.template GetObject<COneForm>(E),
					*ctx.template GetObject<OneForm>(J));
		}
		else if ((Btype == "CTwoForm" || Etype == "COneForm")
				&& Jtype == "COneForm")
		{
			DoMaxwellEq(*ctx.template GetObject<CTwoForm>(B),
					*ctx.template GetObject<COneForm>(E),
					*ctx.template GetObject<COneForm>(J));
		}
		else
		{
			ERROR << "Field type mismatch!!";
		}

	}

	template<typename TE, typename TB, typename TJ>
	void DoMaxwellEq(TB &B, TE & E, TJ const &J)
	{
		B -= Curl(E) * dt;

		E += (Curl(B / mu0) - J) / epsilon0 * dt;
	}

private:
	const Real dt;
	const Real mu0;
	const Real epsilon0;
	const Real speed_of_light;

	std::string E, B, J;
	std::string Etype, Btype, Jtype;

};

} // namespace em_field
} // namespace simpla
#endif  // SRC_EMFIELD_MAXWELL_H_
