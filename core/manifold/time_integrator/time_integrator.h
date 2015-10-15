//
// Created by salmon on 7/7/15.
//

#ifndef SIMPLA_TIME_INTEGRATOR_H
#define SIMPLA_TIME_INTEGRATOR_H
namespace simpla
{
template<typename ...> struct TimeIntegrator;

template<typename TGeo, typename ...Policy>
struct TimeIntegrator<TGeo, Policy...>
{
private:
	typedef TGeo geometry_type;
public:

	TimeIntegrator(TGeo &) { }

	virtual ~TimeIntegrator() { }


	double time() const { return m_time_; }

	void time(double t) { m_time_ = t; }

	double dt() const { return m_dt_; }

	void dt(double p_dt) { m_dt_ = p_dt; }

	virtual void next_time_step() { m_time_ += m_dt_; }

private:
	double m_dt_;
	double m_time_;

};
}// namespace simpla
#endif //SIMPLA_TIME_INTEGRATOR_H
