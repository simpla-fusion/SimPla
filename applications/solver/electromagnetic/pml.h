/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * Maxwell/PML.h
 *
 *  Created on: 2010-12-7
 *      Author: salmon
 */

#include <cmath>
#include <cstddef>
#include <iostream>

#include "../../../src/fetl/fetl.h"
#include "../../../src/utilities/log.h"
#include "../../../src/physics/physical_constants.h"

#include "../../../src/engine/fieldsolver.h"
namespace simpla
{

class LuaObject;

template<typename TM>
class PML: public FieldSolver<TM>
{

	inline Real sigma_(Real r, Real expN, Real dB)
	{
		return (0.5 * (expN + 2.0) * 0.1 * dB * pow(r, expN + 1.0));
	}
	inline Real alpha_(Real r, Real expN, Real dB)
	{
		return (1.0 + 2.0 * pow(r, expN));
	}
public:
	DEFINE_FIELDS (TM)

	typedef Mesh mesh_type;

	Mesh const & mesh;

private:
	Form<1> X10, X11, X12;
	Form<2> X20, X21, X22;

	// alpha
	Form<0> a0, a1, a2;
	// sigma
	Form<0> s0, s1, s2;

	nTuple<6, int> bc_;

	bool isInitilized_;
public:
	PML(mesh_type const & pmesh);
	~PML();

	void Update();
	bool empty() const
	{
		return !isInitilized_;
	}
	void Deserialize(LuaObject const&cfg);
	std::ostream & Serialize(std::ostream & os) const;

	void NextTimeStepE(Real dt, Form<1> const &E1, Form<2> const &B1, Form<1> *dE);
	void NextTimeStepB(Real dt, Form<1> const &E1, Form<2> const &B1, Form<2> *dB);

	void DumpData() const;
};

template<typename TM>
inline std::ostream & operator<<(std::ostream & os, PML<TM> const &self)
{
	return self.Serialize(os);
}

template<typename TM>
PML<TM>::PML(mesh_type const & pmesh) :
		mesh(pmesh),

		a0(pmesh), a1(pmesh), a2(pmesh),

		s0(pmesh), s1(pmesh), s2(pmesh),

		X10(pmesh), X11(pmesh), X12(pmesh),

		X20(pmesh), X21(pmesh), X22(pmesh),

		isInitilized_(false)
{
}

template<typename TM>
PML<TM>::~PML()
{
}

template<typename TM>
void PML<TM>::Update()
{
	isInitilized_ = true;

	nTuple<3, index_type> const &gw = mesh.GetGhostWidth();

	for (int i = 0; i < 3; ++i)
	{

		if (bc_[i * 2] < gw[i] || bc_[i * 2 + 1] < gw[i])
		{
			ERROR << "Illegal configuration! Ghostpoint with is larger than PML points width."

			<< "\n\t[ mesh.GhostWidth = {" << ToString(gw, ",") << "}"

			<< " PML.Width = {" << ToString(bc_, ",") << "} ]";
		}

		else if (bc_[i * 2] + bc_[i * 2 + 1] > 0 && gw[i] < 2)

		{
			ERROR << "Illegal configuration! PML need at least 2 point ghost width."

			<< "\n\t[ mesh.GhostWidth = {" << ToString(gw, ",") << "}"

			<< " PML.Width = {" << ToString(bc_, ",") << "} ]";
		}

	}

	DEFINE_PHYSICAL_CONST(mesh.constants());

	Real dB = 100, expN = 2;

	a0.Fill(1.0);
	a1.Fill(1.0);
	a2.Fill(1.0);
	s0.Fill(0.0);
	s1.Fill(0.0);
	s2.Fill(0.0);
	X10.Fill(0.0);
	X11.Fill(0.0);
	X12.Fill(0.0);
	X20.Fill(0.0);
	X21.Fill(0.0);
	X22.Fill(0.0);

	auto const & dims = mesh.GetDimension();
	auto const & st = mesh.GetStrides();
	auto const & L = mesh.GetExtent();
	Real inv_dx[3];

	for (int i = 0; i < 3; ++i)
	{
		if (dims[i] > 1)
			inv_dx[i] = (L.second[i] - L.first[i]) / static_cast<Real>(dims[i] - 1);
		else
			inv_dx[i] = 0;
	}

	for (index_type ix = 0; ix < dims[0]; ++ix)
		for (index_type iy = 0; iy < dims[1]; ++iy)
			for (index_type iz = 0; iz < dims[2]; ++iz)
			{
				index_type s = ix * st[0] + iy * st[1] + iz * st[2];
				if (ix < bc_[0])
				{
					Real r = static_cast<Real>(bc_[0] - ix) / static_cast<Real>(bc_[0]);
					a0[s] = alpha_(r, expN, dB);
					s0[s] = sigma_(r, expN, dB) * speed_of_light / bc_[0] * inv_dx[0];
				}
				else if (ix > dims[0] - bc_[0 + 1])
				{
					Real r = static_cast<Real>(ix - (dims[0] - bc_[0 + 1])) / static_cast<Real>(bc_[0 + 1]);
					a0[s] = alpha_(r, expN, dB);
					s0[s] = sigma_(r, expN, dB) * speed_of_light / bc_[1] * inv_dx[0];
				};

				if (iy < bc_[2])
				{
					Real r = static_cast<Real>(bc_[2] - iy) / static_cast<Real>(bc_[2]);
					a1[s] = alpha_(r, expN, dB);
					s1[s] = sigma_(r, expN, dB) * speed_of_light / bc_[2] * inv_dx[1];
				}
				else if (iy > dims[1] - bc_[2 + 1])
				{
					Real r = static_cast<Real>(iy - (dims[1] - bc_[2 + 1])) / static_cast<Real>(bc_[2 + 1]);
					a1[s] = alpha_(r, expN, dB);
					s1[s] = sigma_(r, expN, dB) * speed_of_light / bc_[3] * inv_dx[1];
				}

				if (iz < bc_[4])
				{
					Real r = static_cast<Real>(bc_[4] - iz) / static_cast<Real>(bc_[4]);

					a2[s] = alpha_(r, expN, dB);
					s2[s] = sigma_(r, expN, dB) * speed_of_light / bc_[4] * inv_dx[2];
				}
				else if (iz > dims[2] - bc_[4 + 1])
				{
					Real r = static_cast<Real>(iz - (dims[2] - bc_[4 + 1])) / static_cast<Real>(bc_[4 + 1]);

					a2[s] = alpha_(r, expN, dB);
					s2[s] = sigma_(r, expN, dB) * speed_of_light / bc_[5] * inv_dx[2];
				}
			}

}
template<typename TM>
void PML<TM>::Deserialize(LuaObject const&cfg)
{
	if (cfg.empty())
		return;
	cfg["Width"].as(&bc_);
	Update();
	LOGGER << "Load PML solver" << DONE;
}
template<typename TM>

std::ostream & PML<TM>::Serialize(std::ostream & os) const
{
	os << "\tPML={  Width={" << ToString(bc_, ",") << " } }\n";
	return os;
}

template<typename TM>
void PML<TM>::DumpData() const
{
}

template<typename TM>
void PML<TM>::NextTimeStepE(Real dt, Form<1> const&E1, Form<2> const&B1, Form<1> *dE)
{
	LOGGER << "PML push E";
	dE->Fill(0);

	DEFINE_PHYSICAL_CONST(mesh.constants());

	Form<1> dX1(mesh);

	dX1 = (-2.0 * s0 * X10 + CurlPDX(B1 / mu0)) / (a0 / dt + s0);
	X10 += dX1;
	*dE = dX1 / epsilon0;

	dX1 = (-2.0 * s1 * X11 + CurlPDY(B1 / mu0)) / (a1 / dt + s1);
	X11 += dX1;
	*dE += dX1 / epsilon0;

	dX1 = (-2.0 * s2 * X12 + CurlPDZ(B1 / mu0)) / (a2 / dt + s2);
	X12 += dX1;
	*dE += dX1 / epsilon0;
}

template<typename TM>
void PML<TM>::NextTimeStepB(Real dt, Form<1> const &E1, Form<2> const&B1, Form<2> *dB)
{
	LOGGER << "PML Push B";
	dB->Fill(0);

	DEFINE_PHYSICAL_CONST(mesh.constants());

	Form<2> dX2(mesh);

	dX2 = (-2.0 * s0 * X20 + CurlPDX(E1)) / (a0 / dt + s0);
	X20 += dX2;
	*dB = -dX2;

	dX2 = (-2.0 * s1 * X21 + CurlPDY(E1)) / (a1 / dt + s1);
	X21 += dX2;
	*dB -= dX2;

	dX2 = (-2.0 * s2 * X22 + CurlPDZ(E1)) / (a2 / dt + s2);
	X22 += dX2;
	*dB -= dX2;
}

} //namespace simpla
