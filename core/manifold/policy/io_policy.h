/**
 * @file io_policy.h
 * @author salmon
 * @date 2015-12-02.
 */

#ifndef SIMPLA_IO_POLICY_H
#define SIMPLA_IO_POLICY_H

#include "../../dataset/dataset.h"
#include "../../io/xdmf_stream.h"

namespace simpla
{
namespace manifold { namespace policy
{
/**
 * @ingroup manifold
 */

template<typename ...> struct IOPolicy;

template<typename TGeo>
struct IOPolicy<TGeo> : public io::XDMFStream
{
private:

    typedef TGeo geometry_type;

    typedef IOPolicy<geometry_type> this_type;

    typedef io::XDMFStream base_type;

    typedef typename TGeo::id_type id_type;

    geometry_type const &m_geo_;


public:

    typedef this_type io_policy;


    IOPolicy(geometry_type &geo) :
            m_geo_(geo) { }

    virtual ~IOPolicy() { }

    virtual Real time() const = 0;

    virtual DataSet grid_vertices() const = 0;


    virtual void deploy()
    {
    }

    virtual void next_time_step()
    {
        base_type::next_time_step();
    }

    void set_grid()
    {
        if (m_geo_.topology_type() == "CoRectMesh")
        {
            int ndims = m_geo_.ndims;

            nTuple<size_t, 3> dims;

            dims = m_geo_.dimensions();

            nTuple<Real, 3> xmin, dx;

            std::tie(xmin, std::ignore) = m_geo_.box();

            dx = m_geo_.dx();

            base_type::set_grid(ndims, &dims[0], &xmin[0], &dx[0]);
        }

        else if (m_geo_.topology_type() == "SMesh")
        {
            base_type::set_grid(grid_vertices());
        }
        base_type::set_grid();
    }


    template<typename TDict> void load(TDict const &) { }

    template<typename OS> OS &print(OS &os) const
    {
        os << "\t StoragePolicy={ Default }," << std::endl;
        return os;
    }


};//template<typename TGeo> struct IoPolicy

} //namespace policy
} //namespace manifold

namespace traits
{

template<typename TGeo>
struct type_id<manifold::policy::IOPolicy<TGeo>>
{
    static std::string name()
    {
        return "IOPolicy<" + type_id<TGeo>::name() + ">";
    }
};
}//namespace traits

}//namespace simpla

#endif //SIMPLA_IO_POLICY_H
