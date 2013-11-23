/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * Maxwell/PML.h
 *
 *  Created on: 2010-12-7
 *      Author: salmon
 */

#include "fetl/fetl.h"
#include "mesh/uniform_rect.h"
#include "physics/physical_constants.h"
namespace simpla
{

inline Real sigma_(Real r, Real expN, Real dB)
{
	return (0.5 * (expN + 2.0) * 0.1 * dB * pow(r, expN + 1.0));
}
inline Real alpha_(Real r, Real expN, Real dB)
{
	return (1.0 + 2.0 * pow(r, expN));
}

template<typename TMesh>
class PML
{

public:
	typedef TMesh mesh_type;

	template<int IFORM> using Form = Field<Geometry<mesh_type,IFORM>,Real >;
	template<int IFORM> using VForm = Field<Geometry<mesh_type,IFORM>,nTuple<3,Real> >;

	mesh_type const & mesh;

	const Real mu0;
	const Real epsilon0;
	const Real speed_of_light;

	Form<2> X10, X11, X12;
	Form<1> X20, X21, X22;

	// alpha
	Form<0> a0, a1, a2;
	// sigma
	Form<0> s0, s1, s2;

	nTuple<6, int> bc_;

	template<typename PT>
	PML(mesh_type const & pmesh, const PT & phys) :
			mesh(pmesh),

			mu0(mesh.phys_constants["permeability of free space"]),

			epsilon0(mesh.phys_constants["permittivity of free space"]),

			speed_of_light(mesh.phys_constants["speed of light"]),

			a0(mesh), a1(mesh), a2(mesh),

			s0(mesh), s1(mesh), s2(mesh),

			X10(mesh), X11(mesh), X12(mesh),

			X20(mesh), X21(mesh), X22(mesh)

	{

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

		auto const & dims = mesh.dims_;
		auto const & st = mesh.strides_;

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
								* mesh.inv_dx_[0];
					}
					else if (ix > dims[0] - bc_[0 + 1])
					{
						Real r = static_cast<Real>(ix - (dims[0] - bc_[0 + 1]))
								/ static_cast<Real>(bc_[0 + 1]);
						a0[s] = alpha_(r, expN, dB);
						s0[s] = sigma_(r, expN, dB) * speed_of_light / bc_[1]
								* mesh.inv_dx_[0];
					}

					if (iy < bc_[2])
					{
						Real r = static_cast<Real>(bc_[2] - iy)
								/ static_cast<Real>(bc_[2]);
						a1[s] = alpha_(r, expN, dB);
						s1[s] = sigma_(r, expN, dB) * speed_of_light / bc_[2]
								* mesh.inv_dx_[1];
					}
					else if (iy > dims[1] - bc_[2 + 1])
					{
						Real r = static_cast<Real>(iy - (dims[1] - bc_[2 + 1]))
								/ static_cast<Real>(bc_[2 + 1]);
						a1[s] = alpha_(r, expN, dB);
						s1[s] = sigma_(r, expN, dB) * speed_of_light / bc_[3]
								* mesh.inv_dx_[1];
					}

					if (iz < bc_[4])
					{
						Real r = static_cast<Real>(bc_[4] - iz)
								/ static_cast<Real>(bc_[4]);

						a2[s] = alpha_(r, expN, dB);
						s2[s] = sigma_(r, expN, dB) * speed_of_light / bc_[4]
								* mesh.inv_dx_[2];
					}
					else if (iz > dims[2] - bc_[4 + 1])
					{
						Real r = static_cast<Real>(iz - (dims[2] - bc_[4 + 1]))
								/ static_cast<Real>(bc_[4 + 1]);

						a2[s] = alpha_(r, expN, dB);
						s2[s] = sigma_(r, expN, dB) * speed_of_light / bc_[5]
								* mesh.inv_dx_[2];
					}
				}

	}

	void Eval(Form<1> &E1, Form<2> &B1, Form<1> const &J1, Real dt)
	{
		LOG << "Run module PML";

		Form<1> dX2(mesh);

		dX2 = (-2.0 * s0 * X20 + CurlPDX(B1 / mu0)) / (a0 / dt + s0);
		X20 += dX2;
		E1 += dX2 / epsilon0;

		dX2 = (-2.0 * s1 * X21 + CurlPDY(B1 / mu0)) / (a1 / dt + s1);
		X21 += dX2;
		E1 += dX2 / epsilon0;

		dX2 = (-2.0 * s2 * X22 + CurlPDZ(B1 / mu0)) / (a2 / dt + s2);
		X22 += dX2;
		E1 += dX2 / epsilon0;

		E1 -= J1 / epsilon0 * dt;

		Form<2> dX1(mesh);

		dX1 = (-2.0 * s0 * X10 + CurlPDX(E1)) / (a0 / dt + s0);
		X10 += dX1;
		B1 -= dX1;

		dX1 = (-2.0 * s1 * X11 + CurlPDY(E1)) / (a1 / dt + s1);
		X11 += dX1;
		B1 -= dX1;

		dX1 = (-2.0 * s2 * X12 + CurlPDZ(E1)) / (a2 / dt + s2);
		X12 += dX1;
		B1 -= dX1;

	}
};
} //namespace simpla
