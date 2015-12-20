/**
 * @file ParticleTracker.h
 * @author salmon
 * @date 2015-12-20.
 */

#ifndef SIMPLA_PARTICLETRACKER_H
#define SIMPLA_PARTICLETRACKER_H

namespace simpla { namespace particle
{
template<typename P>
class ParticleTracker : public P::mesh_type::AttributeEntity
{
    typedef ParticleTracker<P> this_type;
    typedef P particle_type;
    typedef P::mesh_type::AttributeEntity base_type;
public:
    ParticleTracker(particle_type *p);

    ParticleTracker(ParticleTracker const &other);

    virtual   ~ParticleTracker();

    virtual bool is_a(std::type_info const &info) const
    {
        return typeid(this_type) == info || base_type::is_a(info);
    }

    virtual std::string get_class_name() const
    {
        return "ParticleTracker<" + mesh().get_class_name() + ">";
    }

    virtual data_model::DataSet data_set() const;

private:
    typedef typename P::point_type value_type;
    typedef std::tuple <size_t, value_type> ele_type;

};

template<typename P, typename M>
ParticleTracker<P, M>::ParticleTracker(const mesh_type &m, const string &s_name)
        : base_type(m, s_name)
{

}

template<typename P, typename M>
ParticleTracker<P, M>::ParticleTracker(const ParticleTracker &other)
        : base_type(other)
{

}

template<typename P, typename M>
ParticleTracker<P, M>::~ParticleTracker()
{

}
}}//namespace simpla { namespace particle{

#endif //SIMPLA_PARTICLETRACKER_H
