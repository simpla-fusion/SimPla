/**
 * @file embedded_policy.h
 * @author salmon
 * @date 2015-12-09.
 */

#ifndef SIMPLA_EMBEDDED_POLICY_H
#define SIMPLA_EMBEDDED_POLICY_H


namespace simpla { namespace manifold { namespace policy
{

template<typename ...> struct EmbeddedPolicy;

template<typename TGeo>
struct EmbeddedPolicy<TGeo>
{
private:

    typedef TGeo geometry_type;

    typedef EmbeddedPolicy<geometry_type> this_type;


    typedef typename TGeo::id_type id_type;

    geometry_type const &m_geo_;


public:

    typedef this_type storage_policy;


    EmbeddedPolicy(geometry_type &geo) : m_geo_(geo) { }

    virtual ~EmbeddedPolicy() { }

    template<typename TDict> void load(TDict const &) { }

    template<typename OS> OS &print(OS &os) const
    {
        os << "\t EmbeddedPolicy={ Default }," << std::endl;
        return os;
    }

    bondary_range()const{}

};


}}}//namespace simpla{namespace manifold { namespace policy

#endif //SIMPLA_EMBEDDED_POLICY_H
