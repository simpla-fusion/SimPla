/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * Maxwell/PML.h
 *
 *  Created on: 2010-12-7
 *      Author: salmon
 */

#include <cmath>
#include <iostream>
#include <string>

#include "../../src/fetl/fetl.h"
#include "../../src/fetl/primitives.h"
#include "../../src/physics/physical_constants.h"
#include "../../src/utilities/log.h"

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
	template<typename ... Args>
	PML(mesh_type const & pmesh, Args const & ...);

	~PML();

	bool empty() const
	{
		return !is_loaded_;
	}

	template<typename TDict, typename ...Others>
	void Load(TDict const &dict, Others const & ...);

	void Load(coordinates_type xmin, coordinates_type xmax);

	void Save(std::string const & path, bool is_verbose) const;

	void NextTimeStepE(Real dt, Form<1> const &E1, Form<2> const &B1, Form<1> *dE);

	void NextTimeStepB(Real dt, Form<1> const &E1, Form<2> const &B1, Form<2> *dB);

};

template<typename TM>
template<typename ... Args>
PML<TM>::PML(mesh_type const & pmesh, Args const & ...args)
		: mesh(pmesh),

		a0(pmesh), a1(pmesh), a2(pmesh),

		s0(pmesh), s1(pmesh), s2(pmesh),

		X10(pmesh), X11(pmesh), X12(pmesh),

		X20(pmesh), X21(pmesh), X22(pmesh),

		is_loaded_(false)
{
	Load(std::forward<Args const &>(args)...);
}

template<typename TM>
PML<TM>::~PML()
{
}

template<typename TM>
template<typename TDict, typename ...Others>
void PML<TM>::Load(TDict const &dict, Others const & ...)
{
	Load(dict["Min"].template as<coordinates_type>(), dict["Max"].template as<coordinates_type>());
}

template<typename TM>
void PML<TM>::Load(coordinates_type xmin, coordinates_type xmax)
{
	LOGGER << "Create PML solver [" << xmin << " , " << xmax << " ]";

	DEFINE_PHYSICAL_CONST;

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

	auto ymin = mesh.GetExtents().first;
	auto ymax = mesh.GetExtents().second;

	for (auto s : mesh.Select(VERTEX))
	{
		coordinates_type x = mesh.GetCoordinates(s);

#define DEF(_N_)                                                                    \
		if (x[_N_] < xmin[_N_])                                                         \
		{                                                                           \
			Real r = (xmin[_N_] - x[_N_]) / (xmin[_N_] - ymin[_N_]);                        \
			a##_N_[s] = alpha_(r, expN, dB);                                            \
			s##_N_[s] = sigma_(r, expN, dB) * speed_of_light / (xmin[_N_] - ymin[_N_]);     \
		}                                                                           \
		else if (x[_N_] > xmax[_N_])                                                    \
		{                                                                           \
			Real r = (x[_N_] - xmax[_N_]) / (ymax[_N_] - xmax[_N_]);                        \
			a##_N_[s] = alpha_(r, expN, dB);                                            \
			s##_N_[s] = sigma_(r, expN, dB) * speed_of_light / (ymax[_N_] - xmax[_N_]);     \
		};

		DEF(0)
		DEF(1)
		DEF(2)
#undef DEF
	}

	is_loaded_ = true;

	LOGGER << DONE;

}

template<typename TM>
void PML<TM>::Save(std::string const & path, bool is_verbose) const
{
	UNIMPLEMENT;
}

template<typename OS, typename TM>
OS &operator<<(OS & os, PML<TM> const& self)
{
	self.Print(os);
	return os;
}

template<typename TM>
void PML<TM>::NextTimeStepE(Real dt, Form<1> const&E1, Form<2> const&B1, Form<1> *dE)
{
	LOGGER << "PML push E";
	DEFINE_PHYSICAL_CONST;

	Form<1> dX1(mesh);

	dX1 = (-2.0 * dt * s0 * X10 + CurlPDX(B1 * speed_of_light2) * dt) / (a0 + s0 * dt);
	X10 += dX1;
	*dE += dX1;

	dX1 = (-2.0 * dt * s1 * X11 + CurlPDY(B1 * speed_of_light2) * dt) / (a1 + s1 * dt);
	X11 += dX1;
	*dE += dX1;

	dX1 = (-2.0 * dt * s2 * X12 + CurlPDZ(B1 * speed_of_light2) * dt) / (a2 + s2 * dt);
	X12 += dX1;
	*dE += dX1;

	LOGGER << DONE;
}

template<typename TM>
void PML<TM>::NextTimeStepB(Real dt, Form<1> const &E1, Form<2> const&B1, Form<2> *dB)
{
	LOGGER << "PML Push B";

	DEFINE_PHYSICAL_CONST;

	Form<2> dX2(mesh);

	dX2 = (-2.0 * dt * s0 * X20 + CurlPDX(E1) * dt) / (a0 + s0 * dt);
	X20 += dX2;
	*dB -= dX2;

	dX2 = (-2.0 * dt * s1 * X21 + CurlPDY(E1) * dt) / (a1 + s1 * dt);
	X21 += dX2;
	*dB -= dX2;

	dX2 = (-2.0 * dt * s2 * X22 + CurlPDZ(E1) * dt) / (a2 + s2 * dt);
	X22 += dX2;
	*dB -= dX2;
	LOGGER << DONE;
}

} //namespace simpla
