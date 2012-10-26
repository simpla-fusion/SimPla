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
	;

	template<typename PT>
	PML(BaseContext & d, const PT & pt) :
			ctx(d),

			grid(ctx.Grid<TG>()),

			dt(ctx.dt),

			mu0(ctx.PHYS_CONSTANTS["permeability_of_free_space"]),

			epsilon0(ctx.PHYS_CONSTANTS["permittivity_of_free_space"]),

			speed_of_light(ctx.PHYS_CONSTANTS["speed_of_light"]),

			B1(ctx.template GetObject < TwoForm > ("B1")),

			E1(ctx.template GetObject < OneForm > ("E1")),

			J1(ctx.template GetObject < OneForm > ("J1")),

			a0(grid), a1(grid), a2(grid),

			s0(grid), s1(grid), s2(grid),

			X10(grid), X11(grid), X12(grid),

			X20(grid), X21(grid), X22(grid),

			bc_(pt.template get<nTuple<SIX, int> >("bc"))
	{
		Initialize();
	}

	virtual ~PML();
	virtual void Eval();
	void Initialize();
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
