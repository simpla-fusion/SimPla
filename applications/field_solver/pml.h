/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * Maxwell/PML.h
 *
 * \date  2010-12-7
 *      \author  salmon
 */

#include <cmath>
#include <iostream>
#include <string>

#include "../../core/field/field.h"
#include "../../core/utilities/primitives.h"
#include "../../core/utilities/log.h"
#include "../../core/physics/physical_constants.h"

namespace simpla
{

/**
 *  @ingroup FieldSolver
 *  \brief absorb boundary condition, PML
 */
template<typename TM>
class PML
{

	inline Real sigma_(Real r, Real expN, Real dB)
	{
		return (0.5 * (expN + 2.0) * 0.1 * dB * std::pow(r, expN + 1.0));
	}
	inline Real alpha_(Real r, Real expN, Real dB)
	{
		return (1.0 + 2.0 * std::pow(r, expN));
	}
public:

	typedef TM mesh_type;
	typedef typename mesh_type::scalar_type scalar_type;
	typedef typename mesh_type::coordinates_type coordinates_type;

	mesh_type const & mesh;

private:
	typename mesh_type:: template field<EDGE, scalar_type> X10, X11, X12;
	typename mesh_type:: template field<FACE, scalar_type> X20, X21, X22;

	// alpha
	typename mesh_type:: template field<VERTEX, Real> a0, a1, a2;
	// sigma
	typename mesh_type:: template field<VERTEX, Real> s0, s1, s2;

	bool is_loaded_;
public:
	template<typename ... Args>
	PML(mesh_type const & pmesh, Args && ...);

	~PML();

	bool empty() const
	{
		return !is_loaded_;
	}

	template<typename TDict, typename ...Others>
	void load(TDict const &dict, Others const & ...);

	void load(coordinates_type xmin, coordinates_type xmax);

	void save(std::string const & path, bool is_verbose) const;

	void next_timestepE(Real dt,
			typename mesh_type:: template field<EDGE, scalar_type> const &E1,
			typename mesh_type:: template field<FACE, scalar_type> const &B1,
			typename mesh_type:: template field<EDGE, scalar_type> *dE);

	void next_timestepB(Real dt,
			typename mesh_type:: template field<EDGE, scalar_type> const &E1,
			typename mesh_type:: template field<FACE, scalar_type> const &B1,
			typename mesh_type:: template field<FACE, scalar_type> *dB);

};

template<typename TM>
template<typename ... Args>
PML<TM>::PML(mesh_type const & pmesh, Args && ...args) :
		mesh(pmesh),

		a0(pmesh), a1(pmesh), a2(pmesh),

		s0(pmesh), s1(pmesh), s2(pmesh),

		X10(pmesh), X11(pmesh), X12(pmesh),

		X20(pmesh), X21(pmesh), X22(pmesh),

		is_loaded_(false)
{
	load(std::forward<Args >(args)...);
}

template<typename TM>
PML<TM>::~PML()
{
}

template<typename TM>
template<typename TDict, typename ...Others>
void PML<TM>::load(TDict const &dict, Others const & ...)
{
	load(dict["Min"].template as<coordinates_type>(),
			dict["Max"].template as<coordinates_type>());
}

template<typename TM>
void PML<TM>::load(coordinates_type xmin, coordinates_type xmax)
{
	LOGGER << "create PML solver [" << xmin << " , " << xmax << " ]";

	DEFINE_PHYSICAL_CONST;

	Real dB = 100, expN = 2;

	a0.fill(1.0);
	a1.fill(1.0);
	a2.fill(1.0);
	s0.fill(0.0);
	s1.fill(0.0);
	s2.fill(0.0);
	X10.fill(0.0);
	X11.fill(0.0);
	X12.fill(0.0);
	X20.fill(0.0);
	X21.fill(0.0);
	X22.fill(0.0);

	auto ymin = mesh.get_extents().first;
	auto ymax = mesh.get_extents().second;

	for (auto s : mesh.select(VERTEX))
	{
		coordinates_type x = mesh.get_coordinates(s);

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
void PML<TM>::save(std::string const & path, bool is_verbose) const
{
	UNIMPLEMENTED;
}

template<typename OS, typename TM>
OS &operator<<(OS & os, PML<TM> const& self)
{
	self.print(os);
	return os;
}

template<typename TM>
void PML<TM>::next_timestepE(Real dt,
		typename mesh_type:: template field<EDGE, scalar_type> const&E1,
		typename mesh_type:: template field<FACE, scalar_type> const&B1,
		typename mesh_type:: template field<EDGE, scalar_type> *dE)
{
	LOGGER << "PML push E";

	DEFINE_PHYSICAL_CONST

	auto dX1 = mesh.template make_field<EDGE, scalar_type>();

	dX1 = (-2.0 * dt * s0 * X10 + CurlPDX(B1) / (mu0 * epsilon0) * dt)
			/ (a0 + s0 * dt);
	X10 += dX1;
	*dE += dX1;

	dX1 = (-2.0 * dt * s1 * X11 + CurlPDY(B1) / (mu0 * epsilon0) * dt)
			/ (a1 + s1 * dt);
	X11 += dX1;
	*dE += dX1;

	dX1 = (-2.0 * dt * s2 * X12 + CurlPDZ(B1) / (mu0 * epsilon0) * dt)
			/ (a2 + s2 * dt);
	X12 += dX1;
	*dE += dX1;

	LOGGER << DONE;
}

template<typename TM>
void PML<TM>::next_timestepB(Real dt,
		typename mesh_type:: template field<EDGE, scalar_type> const &E1,
		typename mesh_type:: template field<FACE, scalar_type> const&B1,
		typename mesh_type:: template field<FACE, scalar_type> *dB)
{
	LOGGER << "PML Push B";

	DEFINE_PHYSICAL_CONST

	auto dX2 = mesh.template make_field<FACE, scalar_type>();

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
