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


	double time() const
	{
		return m_time_;
	}

	void time(double t)
	{
		m_time_ = t;
	}

	double m_time_;

};
}// namespace simpla
#endif //SIMPLA_TIME_INTEGRATOR_H
