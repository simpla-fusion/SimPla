/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 *
 *  Created on: 2011-3-22
 *      Author: salmon
 *
 *  cyclebc.cpp
 * 
 */
#include "engine/local_comm.h"
#include <string>
#include <algorithm>
#include <list>
#include "defs.h"
#include "engine/context.h"

LocalComm::LocalComm(Context::Holder ctx) :
		ctx_(ctx)
{
	Grid const & grid = *ctx_->grid;
	for (int i = 0; i < 3; ++i)
	{
		if (grid.dims[i] > 1 && ctx_->bc[i * 2] < 0)
		{
			SizeType L = grid.dims[i] - grid.ghostWidth[i] * 2 - 1;
			if (L > 0)
			{
				IVec3 offset;
				offset = 0;
				offset[i] = -L;
				neighour.push_back(offset);
				offset[i] = L;
				neighour.push_back(offset);
			}
		}
		if (grid.dims[(i + 1) % 3] > 1 && grid.dims[(i + 2) % 3] > 1
				&& ctx_->bc[((i + 1) % 3) * 2] < 0 //
		&& ctx_->bc[((i + 2) % 3) * 2] < 0)
		{
			int ix = (i + 1) % 3;
			int iy = (i + 2) % 3;
			SizeType Lx = grid.dims[ix] - grid.ghostWidth[ix] * 2 - 1;
			SizeType Ly = grid.dims[iy] - grid.ghostWidth[iy] * 2 - 1;
			Lx = (Lx < 0) ? 0 : Lx;
			Ly = (Ly < 0) ? 0 : Ly;

			if (Lx + Ly > 0)
			{
				IVec3 offset;
				offset = 0;
				offset[ix] = -Lx;
				offset[iy] = -Ly;
				neighour.push_back(offset);
				offset[ix] = -Lx;
				offset[iy] = Ly;
				neighour.push_back(offset);
				offset[ix] = Lx;
				offset[iy] = -Ly;
				neighour.push_back(offset);
				offset[ix] = Lx;
				offset[iy] = Ly;
				neighour.push_back(offset);
			}
		}
	}
	if (grid.dims[0] > 1 && grid.dims[1] > 1 && grid.dims[2] > 1
			&& ctx_->bc[0 * 2] < 0 && ctx_->bc[1 * 2] < 0
			&& ctx_->bc[2 * 2] < 0)
	{
		SizeType Lx = grid.dims[0] - grid.ghostWidth[0] * 2 - 1;
		SizeType Ly = grid.dims[1] - grid.ghostWidth[1] * 2 - 1;
		SizeType Lz = grid.dims[2] - grid.ghostWidth[2] * 2 - 1;
		Lx = (Lx < 0) ? 0 : Lx;
		Ly = (Ly < 0) ? 0 : Ly;
		Lz = (Lz < 0) ? 0 : Lz;

		if (Lx + Ly + Lz > 0)
		{
			IVec3 offset;
			offset[0] = -Lx;
			offset[1] = -Ly;
			offset[2] = -Ly;
			neighour.push_back(offset);
			offset[0] = -Lx;
			offset[1] = -Ly;
			offset[2] = Ly;
			neighour.push_back(offset);
			offset[0] = -Lx;
			offset[1] = Ly;
			offset[2] = -Ly;
			neighour.push_back(offset);
			offset[0] = -Lx;
			offset[1] = Ly;
			offset[2] = Ly;
			neighour.push_back(offset);
			offset[0] = Lx;
			offset[1] = -Ly;
			offset[2] = -Ly;
			neighour.push_back(offset);
			offset[0] = Lx;
			offset[1] = -Ly;
			offset[2] = Ly;
			neighour.push_back(offset);
			offset[0] = Lx;
			offset[1] = Ly;
			offset[2] = -Ly;
			neighour.push_back(offset);
			offset[0] = Lx;
			offset[1] = Ly;
			offset[2] = Ly;
			neighour.push_back(offset);
		}
	}
}

void LocalComm::updateField(const std::string& name)
{
	DINGDONG;
	Context::FieldMap::iterator it = ctx_->fields.find(name);
	if (it == ctx_->fields.end())
	{
		WARNING("Can not find field ["+name+"]!");
		return;
	}

	int ncomp = it->second->getDimensions()[3];

	Scalar* lhs = reinterpret_cast<Scalar*>(it->second->getData());

//IVec3 	idx;
//	IVec3 st(ctx_->grid->strides);
//
//	for (int d0 = 0; d0 < 3; ++d0)
//	{
//		if (ctx_->grid->dims[d0] - 2 * ctx_->grid->ghostWidth[d0] <= 0
//				|| (ctx_->bc[d0 * 2] <0 && ctx_->bc[d0 * 2 + 1] <0))
//		{
//			continue;
//		}
//		SizeType gw = ctx_->grid->ghostWidth[d0];
//		SizeType lw = ctx_->grid->dims[d0];
//		int d1 = (d0 + 1) % 3;
//		int d2 = (d0 + 2) % 3;
//
//		for (idx[d1] = 0; idx[d1] < ctx_->grid->dims[d1]; ++idx[d1])
//			for (idx[d2] = 0; idx[d2] < ctx_->grid->dims[d2]; ++idx[d2])
//			{
//				SizeType rs = (gw) * st[d0] + idx[d1] * st[d1]
//						+ idx[d2] * st[d2];
//
//				for (idx[d0] = gw - 1; idx[d0] >= 0; --idx[d0])
//				{
//					SizeType ls = idx[0] * st[0] + idx[1] * st[1]
//							+ idx[2] * st[2];
//					for (int s = 0; s < ncomp; ++s)
//					{
//						lhs[ncomp * (ls) + s] = lhs[ncomp * (rs) + s];
//					}
//				}
//
//				rs = (lw - gw - 1) * st[d0] + idx[d1] * st[d1]
//						+ idx[d2] * st[d2];
//
//				for (idx[d0] = lw - gw; idx[d0] < lw; ++idx[d0])
//				{
//					SizeType ls = idx[0] * st[0] + idx[1] * st[1]
//							+ idx[2] * st[2];
//					for (int s = 0; s < ncomp; ++s)
//					{
//						lhs[ncomp * (ls) + s] = lhs[ncomp * (rs) + s];
//					}
//
//				}
//			}
//	}

	Scalar * rhs = lhs;

	for (std::list<IVec3>::iterator nb = neighour.begin(); nb != neighour.end();
			++nb)
	{
		IVec3 lmin, lmax, rmin, rmax, start, count;

		// Check overlap of center area ------------------------

		lmin = 0;
		lmax = ctx_->grid->dims;
		rmin = *nb;
		rmax = rmin + ctx_->grid->dims - ctx_->grid->ghostWidth * 2;

		for (int i = 0; i < 3; ++i)
		{
			start[i] = std::max(lmin[i], rmin[i]);
			count[i] = std::min(lmax[i], rmax[i]) - start[i];
		}
		IVec3 const &ls = ctx_->grid->strides;

		IVec3 const &rs = ctx_->grid->strides;

		IVec3 I;

		for (I[0] = 0; I[0] < count[0]; ++I[0])
			for (I[1] = 0; I[1] < count[1]; ++I[1])
				for (I[2] = 0; I[2] < count[2]; ++I[2])
				{

					SizeType lI = Dot(I + start - lmin, ls);
					SizeType rI = Dot(I + start - rmin, rs);

					for (int s = 0; s < ncomp; ++s)
					{
						lhs[lI * ncomp + s] = rhs[rI * ncomp + s];
					}
				}
	}
	DONE

}
