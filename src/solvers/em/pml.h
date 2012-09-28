/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * Maxwell/PML.h
 *
 *  Created on: 2010-12-7
 *      Author: salmon
 */

#ifndef SRC_EMFIELD_PML_H_
#define SRC_EMFIELD_PML_H_
#include "fetl/fetl.h"
#include "fetl/vector_calculus.h"
#include "engine/solver.h"
#include "engine/context.h"

namespace simpla
{
namespace em
{

using namespace fetl;
template<typename TV, typename TG>
class PML
{
public:

	typedef TG Grid;
	typedef PML<TV, TG> ThisType;
	typedef TR1::shared_ptr<ThisType> Holder;

	DEFINE_FIELDS(TV,TG);

	explicit PML(Context::Holder ctx, const ptree & properties) :
	ctx_(ctx), grid(ctx->getGrid<Grid>()), bc_(properties.get< nTuple<SIX, int> >("bc"))
	{
		PreProcess();
	}

	virtual ~PML()
	{
	}

	virtual void PreProcess();
	virtual void Process();
	virtual void PostProcess();
private:
	Context::Holder ctx_;
	TG const & grid;
	nTuple<SIX, int> bc_;
	Real dt;
	double mu0;
	double epsilon0;
	double speed_of_light;

	TR1::shared_ptr<TwoForm> X10, X11, X12;
	TR1::shared_ptr<OneForm> X20, X21, X22;

	// alpha
	TR1::shared_ptr<RScalarField> a0, a1, a2;
	// sigma
	TR1::shared_ptr<RScalarField> s0, s1, s2;

	TR1::shared_ptr<OneForm> E1;
	TR1::shared_ptr<TwoForm> B1;
	TR1::shared_ptr<OneForm> J1;

};

inline Real sigma_(Real r, Real expN, Real dB)
{
	return (0.5 * (expN + 2.0) * 0.1 * dB * pow(r, expN + 1.0));
}
inline Real alpha_(Real r, Real expN, Real dB)
{
	return (1.0 + 2.0 * pow(r, expN));
}
template<typename TV, typename TG>
void PML<TV, TG>::PreProcess()
{
//	dt = ctx_->grid.dt;
//	mu0 = ctx_->permeability_of_free_space;
//	epsilon0 = ctx_->permittivity_of_free_space;

	speed_of_light = 1.0 / sqrt(mu0 * epsilon0);

	a0 = ctx_->template GetObject<ZeroForm>("");
	a1 = ctx_->template GetObject<ZeroForm>("");
	a2 = ctx_->template GetObject<ZeroForm>("");

	s0 = ctx_->template GetObject<ZeroForm>("");
	s1 = ctx_->template GetObject<ZeroForm>("");
	s2 = ctx_->template GetObject<ZeroForm>("");

	X10 = ctx_->template GetObject<TwoForm>("X10");

	X11 = ctx_->template GetObject<TwoForm>("X11");

	X12 = ctx_->template GetObject<TwoForm>("X12");

	X20 = ctx_->template GetObject<OneForm>("X20");

	X21 = ctx_->template GetObject<OneForm>("X21");

	X22 = ctx_->template GetObject<OneForm>("X22");

	E1 = ctx_->template GetObject<OneForm>("E1");

	B1 = ctx_->template GetObject<TwoForm>("B1");

	J1 = ctx_->template GetObject<OneForm>("J1");

	Real dB = 100, expN = 2;

	*a0 = 1.0;
	*a1 = 1.0;
	*a2 = 1.0;
	*s0 = 0.0;
	*s1 = 0.0;
	*s2 = 0.0;
	*X10 = 0.0;
	*X11 = 0.0;
	*X12 = 0.0;
	*X20 = 0.0;
	*X21 = 0.0;
	*X22 = 0.0;

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
					(*a0)[s] = alpha_(r, expN, dB);
					(*s0)[s] = sigma_(r, expN, dB) * speed_of_light / bc_[0]
							* grid.inv_dx[0];
				}
				else if (ix > dims[0] - bc_[0 + 1])
				{
					Real r = static_cast<Real>(ix - (dims[0] - bc_[0 + 1]))
							/ static_cast<Real>(bc_[0 + 1]);
					(*a0)[s] = alpha_(r, expN, dB);
					(*s0)[s] = sigma_(r, expN, dB) * speed_of_light / bc_[1]
							* grid.inv_dx[0];
				}

				if (iy < bc_[2])
				{
					Real r = static_cast<Real>(bc_[2] - iy)
							/ static_cast<Real>(bc_[2]);
					(*a1)[s] = alpha_(r, expN, dB);
					(*s1)[s] = sigma_(r, expN, dB) * speed_of_light / bc_[2]
							* grid.inv_dx[1];
				}
				else if (iy > dims[1] - bc_[2 + 1])
				{
					Real r = static_cast<Real>(iy - (dims[1] - bc_[2 + 1]))
							/ static_cast<Real>(bc_[2 + 1]);
					(*a1)[s] = alpha_(r, expN, dB);
					(*s1)[s] = sigma_(r, expN, dB) * speed_of_light / bc_[3]
							* grid.inv_dx[1];
				}

				if (iz < bc_[4])
				{
					Real r = static_cast<Real>(bc_[4] - iz)
							/ static_cast<Real>(bc_[4]);

					(*a2)[s] = alpha_(r, expN, dB);
					(*s2)[s] = sigma_(r, expN, dB) * speed_of_light / bc_[4]
							* grid.inv_dx[2];
				}
				else if (iz > dims[2] - bc_[4 + 1])
				{
					Real r = static_cast<Real>(iz - (dims[2] - bc_[4 + 1]))
							/ static_cast<Real>(bc_[4 + 1]);

					(*a2)[s] = alpha_(r, expN, dB);
					(*s2)[s] = sigma_(r, expN, dB) * speed_of_light / bc_[5]
							* grid.inv_dx[2];
				}
			}

}

template<typename TV, typename TG>
void PML<TV, TG>::PostProcess()
{
}

template<typename TV, typename TG>
void PML<TV, TG>::Process()
{
	using namespace vector_calculus;

//	TwoForm& dX1 = *ctx_->template GetObject<TwoForm>();

	TwoForm dX1(grid);

	dX1 = (-2.0 * (*s0) * (*X10) + CurlPD<0>((*E1))) / ((*a0) / dt + (*s0));
	(*X10) += dX1;
	(*B1) -= dX1;

	dX1 = (-2.0 * (*s1) * (*X11) + CurlPD<1>((*E1))) / ((*a1) / dt + (*s1));
	(*X11) += dX1;
	(*B1) -= dX1;

	dX1 = (-2.0 * (*s2) * (*X12) + CurlPD<2>((*E1))) / ((*a2) / dt + (*s2));

	(*X12) += dX1;
	(*B1) -= dX1;

	//	OneForm &dX2 = *ctx_->template GetObject<OneForm>();
	OneForm dX2(grid);

	dX2 = (-2.0 * (*s0) * (*X20) + CurlPD<0>((*B1) / mu0))
			/ ((*a0) / dt + (*s0));
	(*X20) += dX2;
	(*E1) += dX2 / epsilon0;

	dX2 = (-2.0 * (*s1) * (*X21) + CurlPD<1>((*B1) / mu0))
			/ ((*a1) / dt + (*s1));
	(*X21) += dX2;
	(*E1) += dX2 / epsilon0;

	dX2 = (-2.0 * (*s2) * (*X22) + CurlPD<2>((*B1) / mu0))
			/ ((*a2) / dt + (*s2));
	(*X22) += dX2;
	(*E1) += dX2 / epsilon0;

	(*E1) -= (*J1) / epsilon0 * dt;

}
} // namespace electromagnetic
} // namespace simpla
#endif  // SRC_EMFIELD_PML_H_
