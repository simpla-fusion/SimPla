/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * Maxwell/PML.h
 *
 *  Created on: 2010-12-7
 *      Author: salmon
 */

#ifndef SRC_EMFIELD_PML_H_
#define SRC_EMFIELD_PML_H_
#include "include/simpla_defs.h"
#include "fetl/fetl.h"
#include "engine/context.h"
#include "engine/basemodule.h"
#include "utilities/properties.h"
#include "modules/basemodule.h"
namespace simpla
{
namespace em
{

template<typename TG>
class PML: public BaseModule
{
public:

	typedef PML<TG> ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	DEFINE_FIELDS(TG)

	PML(Context<TG> * d, const PTree & pt);
	virtual ~PML();
	static TR1::function<void()> Create(Context<TG> * d, const PTree & pt)
	{
		return TR1::bind(&ThisType::Eval,
				TR1::shared_ptr<ThisType>(new ThisType(d, pt)));
	}
	virtual void Eval();
private:
	Context<TG> & ctx;
	Grid const & grid;

	const Real dt;
	const Real mu0;
	const Real epsilon0;
	const Real speed_of_light;

	TwoForm X10, X11, X12;
	OneForm X20, X21, X22;

// alpha
	RScalarField a0, a1, a2;
// sigma
	RScalarField s0, s1, s2;

	std::string E, B, J;
	std::string Etype, Btype, Jtype;

	nTuple<SIX, int> bc_;
	// internal variable name, type, reference name

};

} // namespace em
} // namespace simpla
#endif  // SRC_EMFIELD_PML_H_
