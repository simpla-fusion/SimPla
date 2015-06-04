/*Copyright (C) 2007-2011 YU Zhi. All rights reserved.
 * $Id$
 * Maxwell/PML.h
 *
 * \date  2010-12-7
 *      \author  salmon
 */

#include <cmath>
#include <cstdbool>
#include <string>

#include "../../core/gtl/primitives.h"
#include "../../core/mesh/mesh.h"
#include "../../core/physics/physical_constants.h"
#include "../../core/utilities/utilities.h"
#include "../../core/field/field.h"

namespace simpla
{

/**
 *  @ingroup FieldSolver
 *  @brief absorb boundary condition, PML
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
	typedef typename mesh_type::coordinate_tuple coordinate_tuple;

	std::shared_ptr<const mesh_type> m_mesh_;

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
	void init(TDict const &dict, Others const & ...);

	void extents(coordinate_tuple xmin, coordinate_tuple xmax);

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
PML<TM>::PML(mesh_type const & mesh, Args && ...args)
		: m_mesh_(mesh.shared_from_this()),

		a0(*m_mesh_), a1(*m_mesh_), a2(*m_mesh_),

		s0(*m_mesh_), s1(*m_mesh_), s2(*m_mesh_),

		X10(*m_mesh_), X11(*m_mesh_), X12(*m_mesh_),

		X20(*m_mesh_), X21(*m_mesh_), X22(*m_mesh_),

		is_loaded_(false)
{
	init(std::forward<Args >(args)...);
}

template<typename TM>
PML<TM>::~PML()
{
}

template<typename TM>
template<typename TDict, typename ...Others>
void PML<TM>::init(TDict const &dict, Others const & ...)
{
	extents(dict["Min"].template as<coordinate_tuple>(),
			dict["Max"].template as<coordinate_tuple>());
}

template<typename TM>
void PML<TM>::extents(coordinate_tuple xmin, coordinate_tuple xmax)
{
	LOGGER << "create PML solver [" << xmin << " , " << xmax << " ]";

	DEFINE_PHYSICAL_CONST

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

	coordinate_tuple ymin, ymax;
	std::tie(ymin, ymax) = m_mesh_->extents();

	for (auto s : m_mesh_->template domain<VERTEX>())
	{
		coordinate_tuple x = m_mesh_->coordinates(s);

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
void PML<TM>::next_timestepE(Real dt,
		typename mesh_type:: template field<EDGE, scalar_type> const&E1,
		typename mesh_type:: template field<FACE, scalar_type> const&B1,
		typename mesh_type:: template field<EDGE, scalar_type> *dE)
{
	VERBOSE << "PML push E" << endl;

	DEFINE_PHYSICAL_CONST

	auto dX1 = m_mesh_->template make_form<EDGE, scalar_type>();

	dX1 = (-2.0 * dt * s0 * X10 + curl_pdx(B1) / (mu0 * epsilon0) * dt)
			/ (a0 + s0 * dt);
	X10 += dX1;
	*dE += dX1;

	dX1 = (-2.0 * dt * s1 * X11 + curl_pdy(B1) / (mu0 * epsilon0) * dt)
			/ (a1 + s1 * dt);
	X11 += dX1;
	*dE += dX1;

	dX1 = (-2.0 * dt * s2 * X12 + curl_pdz(B1) / (mu0 * epsilon0) * dt)
			/ (a2 + s2 * dt);
	X12 += dX1;
	*dE += dX1;
}

template<typename TM>
void PML<TM>::next_timestepB(Real dt,
		typename mesh_type:: template field<EDGE, scalar_type> const&E1,
		typename mesh_type:: template field<FACE, scalar_type> const&B1,
		typename mesh_type:: template field<FACE, scalar_type> *dB)
{
	VERBOSE << "PML Push B" << endl;

	DEFINE_PHYSICAL_CONST

	auto dX2 = m_mesh_->template make_form<FACE, scalar_type>();

	dX2 = (-2.0 * dt * s0 * X20 + curl_pdx(E1) * dt) / (a0 + s0 * dt);
	X20 += dX2;
	*dB -= dX2;

	dX2 = (-2.0 * dt * s1 * X21 + curl_pdy(E1) * dt) / (a1 + s1 * dt);
	X21 += dX2;
	*dB -= dX2;

	dX2 = (-2.0 * dt * s2 * X22 + curl_pdz(E1) * dt) / (a2 + s2 * dt);
	X22 += dX2;
	*dB -= dX2;

}

} //namespace simpla
