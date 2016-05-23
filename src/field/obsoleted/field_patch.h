/**
 * @file field_patch.h
 * @author salmon
 * @date 2015-11-19.
 */

#ifndef SIMPLA_FIELD_PATCH_H
#define SIMPLA_FIELD_PATCH_H


#include <algorithm>
#include <cstdbool>
#include <memory>
#include <string>
#include "../dataset/dataset.h"

namespace simpla
{
template<typename ...> class Field;

template<typename ...> class FieldAMRPolicy;

template<typename TF>
class FieldAMRPolicy : public mesh::EnablePatchFromThis<TF>
{
    typedef TF field_type;
    typedef typename field_type::value_type value_type;
    typedef typename field_type::mesh_type mesh_type;

    static constexpr int iform = field_type::iform;

    virtual mesh_type const &mesh() = 0;

    virtual void deploy() = 0;

    virtual void clear() = 0;

    virtual mesh_type const &mesh() const = 0;

    virtual DataSet const &dataset() const = 0;

    virtual DataSet &dataset() = 0;


    field_type patch(size_t id)
    {

        auto m = mesh().patch(id);

        auto it = dataset().patches.find(id);

        if (it == dataset().patches.end())
        {
            auto res = dataset().patches.insert(
                    std::make_pair(id, m->template dataset<value_type, iform>()));

            if (!res.second) { THROW_EXCEPTION_RUNTIME_ERROR("can not create field Patch!"); }
            else { it = res.first; }
        }

        return field_type(*m, it->second);
    }

    field_type patch(size_t id) const
    {
        auto m = mesh().patch(id);

        auto it = dataset().patches.find(id);

        if (it == dataset().patches.end())
        {
            THROW_EXCEPTION_RUNTIME_ERROR("try to access an  unexist field Patch!");
        }

        return field_type(*m, it->second);
    }

    virtual void erase_patch(size_t id)
    {
        dataset().patches.erase(id);
    }

    virtual void refinement(size_t id)
    {
        auto f_patch = patch(id);

    }

    virtual void coarsen(size_t id)
    {

    }
};
}//namespace simpla

#endif //SIMPLA_FIELD_PATCH_H
