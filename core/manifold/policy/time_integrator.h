/**
 * @file time_integrator.h
 * @author salmon
 * @date 2015-10-13.
 */

#ifndef SIMPLA_TIME_INTEGRATOR_H
#define SIMPLA_TIME_INTEGRATOR_H

namespace simpla
{
namespace policy
{
class TimeIntegrator
{
	/**
	 * 	@name  Time
	 *  @{
	 *
	 */

private:
	Real m_time_ = 0;
	Real m_dt_ = 1.0;
	Real m_CFL_ = 0.5;

public:

	void next_time_step() { m_time_ += m_dt_; }

// Time

	Real dt() const { return m_dt_; }

	void dt(Real pdt) { m_dt_ = pdt; }

	void time(Real p_time) { m_time_ = p_time; }

	Real time() const { return m_time_; }

	/** @} */
};
}// namespace policy
}// namespace simpla

#endif //SIMPLA_TIME_INTEGRATOR_H
