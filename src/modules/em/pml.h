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
#include "engine/modules.h"
#include "engine/basecontext.h"
#include "utilities/properties.h"

namespace simpla
{
namespace em
{

template<typename TG>
class PML: public Module
{
public:

	typedef PML<TG> ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	DEFINE_FIELDS(typename TG::ValueType,TG)

	PML(BaseContext & d, const ptree & pt);
	virtual ~PML();
	virtual void Eval();
private:
	BaseContext & ctx;
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

//input
	TR1::shared_ptr<const OneForm> J1;
//output
	TR1::shared_ptr<OneForm> E1;
	TR1::shared_ptr<TwoForm> B1;

	nTuple<SIX, int> bc_;

};

} // namespace em
} // namespace simpla
#endif  // SRC_EMFIELD_PML_H_
