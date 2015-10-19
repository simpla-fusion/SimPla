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

	template<typename TDict>
	void load(TDict const &dict)
	{

		m_dt_ = dict["Geometry.dt"].template as<Real>(1.0);
		m_time_ = dict["Geometry.Time"].template as<Real>(0);
	}

	template<typename OS>
	OS &print(OS &os) const
	{

		os << "\t TimeIntegator = {" << std::endl
				<< "\t\t Type = \"Default\"," << std::endl
				<< "\t\t Time = " << m_time_ << "," << std::endl
				<< "\t\t dt =   " << m_dt_ << "," << std::endl
				<< "\t }, " << std::endl;
		return os;
	}

	void deploy() { }

	void next_time_step() { m_time_ += m_dt_; }

	double time() const { return m_time_; }

	void time(double t) { m_time_ = t; }

	double dt() const { return m_dt_; }

	void dt(double p_dt) { m_dt_ = p_dt; }


private:
	double m_dt_;
	double m_time_;

};


namespace traits
{
template<typename ... T>
struct type_id<TimeIntegrator<T...> >
{
	static std::string name()
	{
		return "TimeIntegrator<" + type_id<T...>::name() + " >";
	}
};
}//namespace traits
}// namespace simpla
#endif //SIMPLA_TIME_INTEGRATOR_H
