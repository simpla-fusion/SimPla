/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * Maxwell/PML.h
 *
 *  Created on: 2010-12-7
 *      Author: salmon
 */

#include "pml.h"

#include "fetl/grid/uniform_rect.h"

#include "engine/context.h"

namespace simpla
{
namespace em
{

template<>
PML<UniformRectGrid>::~PML()
{
}

inline Real sigma_(Real r, Real expN, Real dB)
{
	return (0.5 * (expN + 2.0) * 0.1 * dB * pow(r, expN + 1.0));
}
inline Real alpha_(Real r, Real expN, Real dB)
{
	return (1.0 + 2.0 * pow(r, expN));
}
template<>
PML<UniformRectGrid>::PML(Context<UniformRectGrid> * d, const ptree & pt) :
		BaseModule(d, pt),

		ctx(*d),

		grid(ctx.grid),

		dt(ctx.grid.dt),

		mu0(ctx.PHYS_CONSTANTS["permeability_of_free_space"]),

		epsilon0(ctx.PHYS_CONSTANTS["permittivity_of_free_space"]),

		speed_of_light(ctx.PHYS_CONSTANTS["speed_of_light"]),

		a0(grid), a1(grid), a2(grid),

		s0(grid), s1(grid), s2(grid),

		X10(grid), X11(grid), X12(grid),

		X20(grid), X21(grid), X22(grid),

		bc_(pt.get < nTuple<SIX, int> > ("Arguments.bc"))
{

	LOG << "Create module PML";

	Real dB = 100, expN = 2;

	a0 = 1.0;
	a1 = 1.0;
	a2 = 1.0;
	s0 = 0.0;
	s1 = 0.0;
	s2 = 0.0;
	X10 = 0.0;
	X11 = 0.0;
	X12 = 0.0;
	X20 = 0.0;
	X21 = 0.0;
	X22 = 0.0;

//   0 1 2 3 4 5 6 7 8 9
	IVec3 const & dims = grid.dims;
	IVec3 const & st = grid.strides;

	for (size_t ix = 0; ix < dims[0]; ++ix)
		for (size_t iy = 0; iy < dims[1]; ++iy)
			for (size_t iz = 0; iz < dims[2]; ++iz)
			{
				size_t s = ix * st[0] + iy * st[1] + iz * st[2];
				if (ix < bc_[0])
				{
					Real r = static_cast<Real>(bc_[0] - ix)
							/ static_cast<Real>(bc_[0]);
					a0[s] = alpha_(r, expN, dB);
					s0[s] = sigma_(r, expN, dB) * speed_of_light / bc_[0]
							* grid.inv_dx[0];
				}
				else if (ix > dims[0] - bc_[0 + 1])
				{
					Real r = static_cast<Real>(ix - (dims[0] - bc_[0 + 1]))
							/ static_cast<Real>(bc_[0 + 1]);
					a0[s] = alpha_(r, expN, dB);
					s0[s] = sigma_(r, expN, dB) * speed_of_light / bc_[1]
							* grid.inv_dx[0];
				}

				if (iy < bc_[2])
				{
					Real r = static_cast<Real>(bc_[2] - iy)
							/ static_cast<Real>(bc_[2]);
					a1[s] = alpha_(r, expN, dB);
					s1[s] = sigma_(r, expN, dB) * speed_of_light / bc_[2]
							* grid.inv_dx[1];
				}
				else if (iy > dims[1] - bc_[2 + 1])
				{
					Real r = static_cast<Real>(iy - (dims[1] - bc_[2 + 1]))
							/ static_cast<Real>(bc_[2 + 1]);
					a1[s] = alpha_(r, expN, dB);
					s1[s] = sigma_(r, expN, dB) * speed_of_light / bc_[3]
							* grid.inv_dx[1];
				}

				if (iz < bc_[4])
				{
					Real r = static_cast<Real>(bc_[4] - iz)
							/ static_cast<Real>(bc_[4]);

					a2[s] = alpha_(r, expN, dB);
					s2[s] = sigma_(r, expN, dB) * speed_of_light / bc_[4]
							* grid.inv_dx[2];
				}
				else if (iz > dims[2] - bc_[4 + 1])
				{
					Real r = static_cast<Real>(iz - (dims[2] - bc_[4 + 1]))
							/ static_cast<Real>(bc_[4 + 1]);

					a2[s] = alpha_(r, expN, dB);
					s2[s] = sigma_(r, expN, dB) * speed_of_light / bc_[5]
							* grid.inv_dx[2];
				}
			}

}
template<>
void PML<UniformRectGrid>::Eval()
{
	LOG << "Run module PML";
	TwoForm &B1 = *TR1::dynamic_pointer_cast < TwoForm > (dataset_["B"]);
	OneForm &E1 = *TR1::dynamic_pointer_cast < OneForm > (dataset_["E"]);
	OneForm &J1 = *TR1::dynamic_pointer_cast < OneForm > (dataset_["J"]);

	OneForm dX2(grid);

	dX2 = (-2.0 * s0 * X20 + CurlPD(Int2Type<0>(), B1 / mu0)) / (a0 / dt + s0);
	X20 += dX2;
	E1 += dX2 / epsilon0;

	dX2 = (-2.0 * s1 * X21 + CurlPD(Int2Type<1>(), B1 / mu0)) / (a1 / dt + s1);
	X21 += dX2;
	E1 += dX2 / epsilon0;

	dX2 = (-2.0 * s2 * X22 + CurlPD(Int2Type<2>(), B1 / mu0)) / (a2 / dt + s2);
	X22 += dX2;
	E1 += dX2 / epsilon0;

	E1 -= J1 / epsilon0 * dt;

	TwoForm dX1(grid);

	dX1 = (-2.0 * s0 * X10 + CurlPD(Int2Type<0>(), E1)) / (a0 / dt + s0);
	X10 += dX1;
	B1 -= dX1;

	dX1 = (-2.0 * s1 * X11 + CurlPD(Int2Type<1>(), E1)) / (a1 / dt + s1);
	X11 += dX1;
	B1 -= dX1;

	dX1 = (-2.0 * s2 * X12 + CurlPD(Int2Type<2>(), E1)) / (a2 / dt + s2);
	X12 += dX1;
	B1 -= dX1;

}
} //namespace em
} //namespace simpla
