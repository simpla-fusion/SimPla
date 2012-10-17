/*
 * boundaryCondition.h
 *
 *  Created on: 2011-8-5
 *      Author: salmon
 */

#ifndef MUR_H_
#define MUR_H_
#include "engine/context.h"
#include "utilities/log.h"
namespace simpla
{

namespace em_field
{

template<int IForm>
void boundaryCondition(Module::Holder ctx, std::string const & fname)
{
	/**
	 *   bc <0  Cycle BC
	 *   bc ==0  PEC BC
	 *   bc ==1  Mur ABC
	 *   bc >1   PML ABC pml_width=ghost_width
	 *
	 * */

	DINGDONG;

	Real r = 1.0;

	IVec3 const & WS = ctx->grid->strides;

	Vec3 a;
	a = ctx->SpeedOfLight * ctx->grid->dt * ctx->grid->inv_dx;

	for (int d = 0; d < 3; ++d)
	{
		if (ctx->grid->dims[d] <= 1)
		{
			continue;
		}
		int d0 = d, d1 = (d + 1) % 3, d2 = (d + 2) % 3;

		if (ctx->bc[2 * d0] == 0) // PEC
		{
			Field<IForm, Real> &F = (*ctx->getField<IForm, Real>(fname));

			IVec3 I;
			I[d0] = 0;
			if (fname == "E1")
			{
				for (I[d1] = 0; I[d1] < ctx->grid->dims[d1]; ++I[d1])
					for (I[d2] = 0; I[d2] < ctx->grid->dims[d2]; ++I[d2])
					{
						size_t s = ctx->grid->get_cell_num(I);
						F[s * 3 + d1] = 0;
						F[s * 3 + d2] = 0;
					}
			}
			else if (fname == "B1")
			{

				for (I[d1] = 0; I[d1] < ctx->grid->dims[d1]; ++I[d1])
					for (I[d2] = 0; I[d2] < ctx->grid->dims[d2]; ++I[d2])
					{
						size_t s = ctx->grid->get_cell_num(I);
						F[s * 3 + d0] = 0;
					}
			}
		}
		else if (ctx->bc[2 * d0] == 1) // Mur ABC
		{
			Field<IForm, Real> & F = (*ctx->getField<IForm, Real>(fname));
			Field<IForm, Real> &F1 = (*ctx->getField<IForm, Real>(fname + "_1"));
			Field<IForm, Real> &F2 = (*ctx->getField<IForm, Real>(fname + "_2"));

			IVec3 I =
			{ 0, 0, 0 };

			for (I[d1] = 0; I[d1] < ctx->grid->dims[d1]; ++I[d1])
				for (I[d2] = 0; I[d2] < ctx->grid->dims[d2]; ++I[d2])
				{
					I[d0] = ctx->grid->ghostWidth[d0] - 1;
					size_t s = ctx->grid->get_cell_num(I);
					for (int p = 0; p < 3; ++p)
					{
						F[s * 3 + p] = 2 * F1[s * 3 + p] - F2[s * 3 + p];

						F[s * 3 + p] += -a[d0]
								* (F1[s * 3 + p] - F1[(s + WS[d0]) * 3 + p]);

						F[s * 3 + p] += +a[d0]
								* (F2[s * 3 + p] - F2[(s + WS[d0]) * 3 + p]);

						F[s * 3 + p] += +0.5 * a[d1] * a[d1]
								* (-2 * F1[s * 3 + p] + F1[(s + WS[d1]) * 3 + p]
										+ F1[(s - WS[d1]) * 3 + p]);
						F[s * 3 + p] += +0.5 * a[d2] * a[d2]
								* (-2 * F1[s * 3 + p] + F1[(s + WS[d2]) * 3 + p]
										+ F1[(s - WS[d2]) * 3 + p]);
						;
						F[s * 3 + p] *= r;
					}

					while (I[d0] >= 0)
					{
						size_t s1 = ctx->grid->get_cell_num(I);
						for (int p = 0; p < 3; ++p)
						{
							F[s1 * 3 + p] = F[s * 3 + p];
						}
						--I[d0];
					}
				}
		}

		if (ctx->bc[2 * d0 + 1] == 0) // PEC
		{

			Field<IForm, Real> &F = (*ctx->getField<IForm, Real>(fname));

			IVec3 I =
			{ 0, 0, 0 };
			I[d0] = ctx->grid->dims[d0] - 1;
			if (fname == "E1")
			{

				for (I[d1] = 0; I[d1] < ctx->grid->dims[d1]; ++I[d1])
					for (I[d2] = 0; I[d2] < ctx->grid->dims[d2]; ++I[d2])
					{
						size_t s = ctx->grid->get_cell_num(I);
						F[s * 3 + d1] = 0;
						F[s * 3 + d2] = 0;
					}
			}
			else if (fname == "B1")
			{
				for (I[d1] = 0; I[d1] < ctx->grid->dims[d1]; ++I[d1])
					for (I[d2] = 0; I[d2] < ctx->grid->dims[d2]; ++I[d2])
					{
						size_t s = ctx->grid->get_cell_num(I);
						F[s * 3 + d0] = 0;
					}
			}

		}
		else if (ctx->bc[2 * d0 + 1] == 1) // Mur ABC
		{
			Field<IForm, Real> &F = (*ctx->getField<IForm, Real>(fname));
			Field<IForm, Real> &F1 = (*ctx->getField<IForm, Real>(fname + "_1"));
			Field<IForm, Real> &F2 = (*ctx->getField<IForm, Real>(fname + "_2"));

			IVec3 I =
			{ 0, 0, 0 };

			for (I[d1] = 0; I[d1] < ctx->grid->dims[d1]; ++I[d1])
				for (I[d2] = 0; I[d2] < ctx->grid->dims[d2]; ++I[d2])
				{
					I[d0] = ctx->grid->dims[d0] - ctx->grid->ghostWidth[d0];
					size_t s = ctx->grid->get_cell_num(I);
					for (int p = 0; p < 3; ++p)
					{
						F[s * 3 + p] = 2 * F1[s * 3 + p] - F2[s * 3 + p];

						F[s * 3 + p] += -a[d0]
								* (F1[s * 3 + p] - F1[(s - WS[d0]) * 3 + p]);

						F[s * 3 + p] += +a[d0]
								* (F2[s * 3 + p] - F2[(s - WS[d0]) * 3 + p]);

						F[s * 3 + p] += (-2 * F1[s * 3 + p]
								+ F1[(s + WS[d1]) * 3 + p]
								+ F1[(s - WS[d1]) * 3 + p])
								* (0.5 * a[d1] * a[d1]);

						F[s * 3 + p] += (-2 * F1[s * 3 + p]
								+ F1[(s + WS[d2]) * 3 + p]
								+ F1[(s - WS[d2]) * 3 + p])
								* (0.5 * a[d2] * a[d2]);
						;
						F[s * 3 + p] *= r;
					}

					while (I[d0] < ctx->grid->dims[d0])
					{
						size_t s1 = ctx->grid->get_cell_num(I);
						for (int p = 0; p < 3; ++p)
						{
							F[s1 * 3 + p] = F[s * 3 + p];
						}
						++I[d0];
					}
				}
		}

	}
}
template<int IForm, typename TV>
void zeroBC(Module::Holder ctx, std::string const & fname)
{

	for (int d = 0; d < 3; ++d)
	{
		int d0 = d, d1 = (d + 1) % 3, d2 = (d + 2) % 3;

		Field<IForm, TV> & F = (*ctx->getField<IForm, TV>(fname));

		IVec3 I =
		{ 0, 0, 0 };
		if (ctx->bc[2 * d0] >= 0)
		{
			I[d0] = 0;
			for (I[d1] = 0; I[d1] < ctx->grid->dims[d1]; ++I[d1])
				for (I[d2] = 0; I[d2] < ctx->grid->dims[d2]; ++I[d2])
				{
					size_t s = ctx->grid->get_cell_num(I);
					F[s * 3 + d0] = 0;
					F[s * 3 + d1] = 0;
					F[s * 3 + d2] = 0;
				}
		}
		if (ctx->bc[2 * d0 + 1] >= 0)
		{
			I[d0] = ctx->grid->dims[d0] - 1;
			for (I[d1] = 0; I[d1] < ctx->grid->dims[d1]; ++I[d1])
				for (I[d2] = 0; I[d2] < ctx->grid->dims[d2]; ++I[d2])
				{
					size_t s = ctx->grid->get_cell_num(I);
					F[s * 3 + d0] = 0;
					F[s * 3 + d1] = 0;
					F[s * 3 + d2] = 0;
				}
		}
	}

}
} // namespace EMField

} // namespace simpla

#endif /* MUR_H_ */
