/**
 * @file patch.h
 * @author salmon
 * @date 2015-11-19.
 */

#ifndef SIMPLA_PATCH_H
#define SIMPLA_PATCH_H

#include "../../parallel/parallel.h"
#include "../../dataset/dataset.h"

namespace simpla { namespace manifold { namespace policy
{

template<typename ...> struct PatchPolicy;

template<typename TMesh>
struct PatchPolicy<TMesh>
{

private:
    typedef TMesh mesh_type;
    typedef typename mesh_type::range_type range_type;
    typedef PatchPolicy<mesh_type> this_type;


public:

    typedef this_type patch_policy;

    PatchPolicy(mesh_type &geo) : m_mesh_(geo) { }

    virtual ~PatchPolicy() { }

    template<typename TDict> void load(TDict const &dict) { }


    template<typename OS>
    OS &print(OS &os) const
    {
        os << "\t PatchPolicy ={ Default }," << std::endl;
        return os;
    }

    template<typename TV, int IFORM>
    void coarse_average(DataSet &ds, size_t id) const;

    template<typename TV, int IFORM>
    void fine_interpolate(DataSet &ds, size_t id) const;

private:
    mesh_type &m_mesh_;

    std::map<size_t, std::shared_ptr<mesh_type>> m_patches_;

}; //template<typename TMesh> struct ParallelPolicy

template<typename TMesh> template<typename TV, int IFORM>
void PatchPolicy<TMesh>::coarse_average(DataSet &ds, size_t id) const
{
    mesh_type const &root_level = m_mesh_;
    mesh_type const &level_level = *m_patches_[id];

    DataSet &main = ds;
    DataSet &patch = ds.patches[id];

    parallel::parallel_for(
            root_level.make_range<IFORM>(i_box),
            [&](range_type const &r)
            {
                for (auto const &s:r)
                {
                    main.template get_value<TV>(m_mesh_.hash(s))
                            = average<TV, IFORM>(patch, (s & (~mesh_type::FULL_OVERFLOW_FLAG)) << 1);
                }
            });

}

template<typename TMesh> template<typename TV, int IFORM>
void PatchPolicy<TMesh>::fine_interpolate(DataSet &ds, size_t id) const
{
    mesh_type const &main_mesh = m_mesh_;
    mesh_type const &patch_mesh = *m_patches_[id];

    DataSet &root = ds;
    DataSet &patch = ds.patches[id];

    parallel::parallel_for(
            patch_mesh.template range<IFORM>(),
            [&](range_type const &r)
            {
                for (auto const &s:r)
                {
                    patch.template get_value<TV>(m_mesh_.hash(s))
                            = interpolate<TV, IFORM>(root, (s & (~mesh_type::FULL_OVERFLOW_FLAG)) >> 1);
                }
            });
}

}}}//namespace simpla { namespace manifold { namespace policy

#endif //SIMPLA_PATCH_H
