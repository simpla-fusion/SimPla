/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * Maxwell/PML.h
 *
 *  Created on: 2010-12-7
 *      Author: salmon
 */

#include <cmath>
#include <cstddef>

#include "../../../src/fetl/fetl.h"
#include "../../../src/utilities/log.h"
#include "../../../src/physics/physical_constants.h"

#include "../../../src/engine/fieldsolver.h"
namespace simpla
{

template<typename TM>
class PML
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

	bool is_loaded_;
public:
	PML(mesh_type const & pmesh);

	~PML();

	void Update();

	bool empty() const
	{
		return !is_loaded_;
	}

	template<typename TDict> void Load(TDict const &dict);

	void Load(coordinates_type xmin, coordinates_type xmax);

	std::ostream & Save(std::ostream & os) const;

	void NextTimeStepE(Real dt, Form<1> const &E1, Form<2> const &B1, Form<1> *dE);

	void NextTimeStepB(Real dt, Form<1> const &E1, Form<2> const &B1, Form<2> *dB);

	void DumpData(std::string const &path = "/DumpData") const;
};

template<typename TM>
inline std::ostream & operator<<(std::ostream & os, PML<TM> const &self)
{
	return self.Save(os);
}

template<typename TM>
PML<TM>::PML(mesh_type const & pmesh) :
		mesh(pmesh),

		a0(pmesh), a1(pmesh), a2(pmesh),

		s0(pmesh), s1(pmesh), s2(pmesh),

		X10(pmesh), X11(pmesh), X12(pmesh),

		X20(pmesh), X21(pmesh), X22(pmesh),

		is_loaded_(false)
{
}

template<typename TM>
PML<TM>::~PML()
{
}

template<typename TM>
template<typename TDict>
void PML<TM>::Load(TDict const &dict)
{
	Load(dict["xmin"].template as<coordinates_type>(), dict["xmax"].template as<coordinates_type>());
}

template<typename TM>
void PML<TM>::Load(coordinates_type xmin, coordinates_type xmax)
{
	LOGGER << "Create PML solver ";

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

	auto ymin = mesh.GetExtent().first;
	auto ymax = mesh.GetExtent().second;

	for (auto s : mesh.GetRange(VERTEX))
	{
		coordinates_type x = mesh.GetCoordinates(s);

		for (int n = 0; n < 3; ++n)
		{
			if (x[n] < xmin[n])
			{
				Real r = (xmin[n] - x[n]) / (xmin[n] - ymin[n]);
				a0[s] = alpha_(r, expN, dB);
				s0[s] = sigma_(r, expN, dB) * speed_of_light / (xmin[n] - ymin[n]);
			}
			else if (x[n] > xmax[n])
			{
				Real r = (x[n] - xmax[n]) / (ymax[n] - xmax[n]);
				a0[s] = alpha_(r, expN, dB);
				s0[s] = sigma_(r, expN, dB) * speed_of_light / (ymax[n] - xmax[n]);
			};
		}
	}

	is_loaded_ = true;

	LOGGER << DONE;

}

template<typename TM>
void PML<TM>::Update()
{
}

template<typename TM>
std::ostream & PML<TM>::Save(std::ostream & os) const
{
	UNIMPLEMENT;
	return os;
}

template<typename OS, typename TM>
OS &operator<<(OS & os, PML<TM> const& self)
{
	self.Save(os);
	return os;
}

template<typename TM>
void PML<TM>::DumpData(std::string const &path) const
{
	UNIMPLEMENT;
}

template<typename TM>
void PML<TM>::NextTimeStepE(Real dt, Form<1> const&E1, Form<2> const&B1, Form<1> *dE)
{
	LOGGER << "PML push E" << DONE;
	DEFINE_PHYSICAL_CONST(mesh.constants());

	Form<1> dX1(mesh);

	dX1 = (-2.0 * s0 * X10 + CurlPDX(B1 / mu0)) / (a0 / dt + s0);
	X10 += dX1;
	*dE += dX1 / dt / epsilon0;

	dX1 = (-2.0 * s1 * X11 + CurlPDY(B1 / mu0)) / (a1 / dt + s1);
	X11 += dX1;
	*dE += dX1 / dt / epsilon0;

	dX1 = (-2.0 * s2 * X12 + CurlPDZ(B1 / mu0)) / (a2 / dt + s2);
	X12 += dX1;
	*dE += dX1 / dt / epsilon0;
}

template<typename TM>
void PML<TM>::NextTimeStepB(Real dt, Form<1> const &E1, Form<2> const&B1, Form<2> *dB)
{
	LOGGER << "PML Push B" << DONE;

	DEFINE_PHYSICAL_CONST(mesh.constants());

	Form<2> dX2(mesh);

	dX2 = (-2.0 * s0 * X20 + CurlPDX(E1)) / (a0 / dt + s0);
	X20 += dX2;
	*dB -= dX2 / dt;

	dX2 = (-2.0 * s1 * X21 + CurlPDY(E1)) / (a1 / dt + s1);
	X21 += dX2;
	*dB -= dX2 / dt;

	dX2 = (-2.0 * s2 * X22 + CurlPDZ(E1)) / (a2 / dt + s2);
	X22 += dX2;
	*dB -= dX2 / dt;
}

} //namespace simpla
