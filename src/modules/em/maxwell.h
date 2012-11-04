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
#include "engine/context.h"
#include "utilities/properties.h"
#include "engine/basemodule.h"
namespace simpla
{
namespace em
{

template<typename TG>
class Maxwell: public BaseModule
{
public:

	typedef Maxwell<TG> ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	DEFINE_FIELDS(typename TG::ValueType, TG)

	Context<TG> & ctx;

	Maxwell(Context<TG> * d, ptree const &pt) :
			BaseModule(d, pt),

			ctx(*d),

			dt(ctx.grid.dt),

			mu0(ctx.PHYS_CONSTANTS["permeability_of_free_space"]),

			epsilon0(ctx.PHYS_CONSTANTS["permittivity_of_free_space"]),

			speed_of_light(ctx.PHYS_CONSTANTS["speed_of_light"])
	{
		LOG << "Create module Maxwell";
	}

	virtual ~Maxwell()
	{
	}

	static TR1::function<void()> Create(Context<TG> * d, const ptree & pt)
	{
		return TR1::bind(&ThisType::Eval,
				TR1::shared_ptr<ThisType>(new ThisType(d, pt)));
	}

	virtual void Eval()
	{
		LOG << "Run module Maxwell";

		if (dataset_["E"]->CheckType(typeid(OneForm)))
		{
			DoMaxwellEq(*TR1::dynamic_pointer_cast<TwoForm>(dataset_["B"]),
					*TR1::dynamic_pointer_cast<OneForm>(dataset_["E"]),
					*TR1::dynamic_pointer_cast<OneForm>(dataset_["J"]));
		}
		else if (dataset_["E"]->CheckType(typeid(COneForm)))
		{
			DoMaxwellEq(*TR1::dynamic_pointer_cast<CTwoForm>(dataset_["B"]),
					*TR1::dynamic_pointer_cast<COneForm>(dataset_["E"]),
					*TR1::dynamic_pointer_cast<COneForm>(dataset_["J"]));
		}
		else
		{
			ERROR << "Field type mismatch!!";
		}

	}

	template<typename TE, typename TB, typename TJ>
	void DoMaxwellEq(TB &B, TE & E, TJ const &J)
	{

		E += (Curl(B / mu0) - J) / epsilon0 * dt;

		B -= Curl(E) * dt;
	}

private:
	const Real dt;
	const Real mu0;
	const Real epsilon0;
	const Real speed_of_light;

};

} // namespace em_field
} // namespace simpla
#endif  // SRC_EMFIELD_MAXWELL_H_
