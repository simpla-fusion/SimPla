/**
 * @file io_policy.h
 * @author salmon
 * @date 2015-12-02.
 */

#ifndef SIMPLA_IO_POLICY_H
#define SIMPLA_IO_POLICY_H

#include "../../dataset/dataset.h"

namespace simpla
{
template<typename ...> class Field;
namespace manifold { namespace policy
{
/**
 * @ingroup manifold
 */
struct MeshIOBase
{
private:
    typedef nTuple<Real, 3> point_type;
public:
    MeshIOBase();

    virtual ~MeshIOBase();

    virtual void set_io_time(Real t);

    virtual bool read();

    virtual void write() const;

    virtual void register_dataset(std::string const &name, DataSet const &ds, int IFORM = 0);

    template<typename TV, typename TM, int IFORM>
    void register_dataset(std::string const &name, Field<TV, TM, std::integral_constant<int, IFORM>> const &f)
    {
        register_dataset(name, f.dataset(), IFORM);
    };


    void set_io_prefix(std::string const &prefix = "", std::string const &name = "unnamed");

    void deploy(int ndims, size_t const *dims, Real const *xmin, Real const *dx);

    void deploy(int ndims, size_t const *dims, point_type const *points);

private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

template<typename ...> struct IOPolicy;

template<typename TGeo>
struct IOPolicy<TGeo> : public MeshIOBase
{
private:

    typedef TGeo geometry_type;

    typedef IOPolicy<geometry_type> this_type;


    typedef typename TGeo::id_type id_type;

    geometry_type const &m_geo_;


public:

    typedef this_type storage_policy;


    IOPolicy(geometry_type &geo) :
            m_geo_(geo) { }

    virtual ~IOPolicy() { }


    virtual void deploy()
    {
        int ndims = m_geo_.ndims;

        nTuple<size_t, 3> dims;

        dims = m_geo_.dimensions();

        nTuple<Real, 3> xmin, dx;

        std::tie(xmin, std::ignore) = m_geo_.box();

        dx = m_geo_.dx();

        MeshIOBase::deploy(ndims, &dims[0], &xmin[0], &dx[0]);
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
        return "IoPolicy<" + type_id<TGeo>::name() + ">";
    }
};
}//namespace traits

}//namespace simpla

#endif //SIMPLA_IO_POLICY_H
