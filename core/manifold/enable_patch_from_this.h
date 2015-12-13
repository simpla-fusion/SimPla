/**
 * @file enable_patch_from_this.h
 * @author salmon
 * @date 2015-12-13.
 */

#ifndef SIMPLA_ENABLE_PATCH_FROM_THIS_H
#define SIMPLA_ENABLE_PATCH_FROM_THIS_H

#include <stddef.h>
#include <memory>
#include <map>
#include "../gtl/utilities/log.h"

namespace simpla { namespace mesh
{
class PatchBase
{
public:
    PatchBase() { };

    virtual ~PatchBase() { }

    virtual std::shared_ptr<PatchBase> patch(size_t id) { return patch_(id); };

    virtual std::shared_ptr<const PatchBase> patch(size_t id) const { return patch_(id); };

    virtual void erase_patch(size_t id) = 0;

    virtual void refinement(size_t id) = 0;

    virtual void coarsen(size_t id) = 0;

protected:
    virtual std::shared_ptr<PatchBase> patch_(size_t) = 0;

    virtual std::shared_ptr<const PatchBase> patch_(size_t) const = 0;
};

template<typename TObject>
class EnablePatchFromThis :
        public PatchBase,
        public std::enable_shared_from_this<TObject>
{
public:
    /**
    * @name patch
    * @{
    */
    EnablePatchFromThis() : m_count_(0), m_parent_(nullptr)
    {

    }

    virtual ~EnablePatchFromThis()
    {
        auto ib = m_patches_.begin();
        while (ib != m_patches_.end())
        {
            auto it = ib;
            ++ib;
            erase_patch(it->first);
        }
    }

    virtual std::shared_ptr<TObject> patch(size_t id)
    {

        auto it = m_patches_.find(id);

        if (it == m_patches_.end())
        {
            auto res = m_patches_.insert(std::make_pair(id, TObject()));

            it = res.first;

            res.first->second->m_parent_ = this;

        }

        return it->second;
    }

    virtual std::shared_ptr<const TObject> patch(size_t id) const
    {
        auto it = m_patches_.find(id);

        if (it == m_patches_.end())
        {
            return std::shared_ptr<const TObject>(nullptr);
        }
        else
        {
            return it->second;
        }


    }

private:
    virtual std::shared_ptr<PatchBase> patch_(size_t id)
    {
        return std::dynamic_pointer_cast<PatchBase>(patch(id));
    }

    virtual std::shared_ptr<const PatchBase> patch_(size_t id) const
    {
        return std::dynamic_pointer_cast<const PatchBase>(patch(id));
    }

public:

    virtual std::tuple<size_t, std::shared_ptr<TObject>> new_patch()
    {
        ++m_count_;
        return std::make_tuple(m_count_, patch(m_count_));
    }


    virtual void erase_patch(size_t id)
    {
        m_patches_.erase(id);
    }

    virtual void refinement()
    {
    }

    virtual void coarsen() const
    {
    }

    /**
     * @}
     */

private:
    size_t m_count_ = 0;

    std::weak_ptr<TObject> m_parent_;
    std::map<size_t, std::shared_ptr<TObject>> m_patches_;
};


}} //namespace simpla { namespace mesh

#endif //SIMPLA_ENABLE_PATCH_FROM_THIS_H
