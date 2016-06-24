//
// Created by salmon on 7/7/15.
//

#ifndef SIMPLA_TIME_INTEGRATOR_H
#define SIMPLA_TIME_INTEGRATOR_H
namespace simpla { namespace manifold { namespace policy
{
template<typename ...> struct TimeIntegrator;

template<typename TGeo, typename ...Policy>
struct TimeIntegrator<TGeo, Policy...>
{
private:
    typedef TGeo geometry_type;
    geometry_type &m_geo_;
public:

    typedef TimeIntegrator<TGeo, Policy...> time_inegral_policy;

    TimeIntegrator(TGeo &geo) : m_geo_(geo) { }

    virtual ~TimeIntegrator() { }

    template<typename TDict>
    void load(TDict const &dict)
    {
        DEFINE_PHYSICAL_CONST;
        auto dx = m_geo_.dx();

        Real default_dt = 0.1 * std::sqrt(dot(dx, dx) / speed_of_light2);

        m_dt_ = dict["dt"].template as<Real>(default_dt);
        m_time_ = dict["Time"].template as<Real>(0);
    }

    virtual std::ostream &print(std::ostream &os) const
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

}}}//namespace  namespace policy namespace CoordinateChart
namespace simpla { namespace traits
{
template<typename ... T>
struct type_id<manifold::policy::TimeIntegrator<T...> >
{
    static std::string name()
    {
        return "TimeIntegrator<" + type_id<T...>::name() + " >";
    }
};
}}// namespace simpla
#endif //SIMPLA_TIME_INTEGRATOR_H
