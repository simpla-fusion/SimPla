/**
 * @file Patch.h
 * @author salmon
 * @date 2015-12-13.
 */

#ifndef SIMPLA_PATCH_H
#define SIMPLA_PATCH_H

#include <stddef.h>
#include <memory>
#include <map>
#include <tuple>
#include "../toolbox/nTuple.h"
#include "../toolbox/Log.h"
#include "../sp_def.h"
#include "Attribute.h"

namespace simpla { namespace base
{


template<typename TObject>
class Patch
{
public:
    /**
    * @name Patch
    * @{
    */
    typedef TObject object_type;

    typedef Patch<object_type> patch_policy;

    Patch() { }

    virtual ~Patch() { }


    virtual object_type &self() = 0;

    virtual object_type const &self() const = 0;

    virtual std::shared_ptr<object_type> create_patch(size_t id) = 0;

    virtual void refinement() { }

    virtual void coarsen() { }


    virtual std::tuple<std::shared_ptr<object_type>, bool>
    insert(std::shared_ptr<object_type> p)
    {
        auto id = m_patch_count_;
        ++m_patch_count_;

        auto res = m_patches_.insert(std::make_pair(id, p));
        return std::make_tuple(res.first->second, res.second);
    };

    /**
     *  find Patch[id], if id do not exist then create_patch(id)
     */
    virtual std::shared_ptr<object_type> patch(size_t id)
    {

        std::shared_ptr<object_type> res(nullptr);

        auto it = m_patches_.find(id);
        if (it != m_patches_.end())
        {
            res = it->second;
        }
        else
        {
            res = create_patch(id);
            bool success = false;
            std::tie(it, success) = m_patches_.insert(std::make_pair(id, res));

            if (success) { res = it->second; }
            else { res = nullptr; }

        }

        return res;
    }

    virtual std::shared_ptr<object_type> patch(size_t id) const
    {
        std::shared_ptr<object_type> res(nullptr);

        auto it = m_patches_.find(id);
        if (it == m_patches_.end())
        {
            res = it->second;
        }
        return res;
    }

    /// @}
protected:
    size_t m_patch_count_;
    std::map<size_t, std::shared_ptr<object_type> > m_patches_;
};


}}// namespace simpla{namespace base{

#endif //SIMPLA_PATCH_H
