/**
 * @file patch.h
 * @author salmon
 * @date 2015-12-13.
 */

#ifndef SIMPLA_PATCH_H
#define SIMPLA_PATCH_H

#include <stddef.h>
#include <memory>
#include <map>
#include "../../gtl/utilities/log.h"

namespace simpla { namespace mesh
{
template<typename ...> class Patch;

template<typename ...> class Layout;


class PatchEntity
{
public:
    PatchEntity() { };

    virtual ~PatchEntity() { }

    virtual void refinement() = 0;

    virtual void coarsen() = 0;

    virtual void sync_patches() { };

    virtual std::shared_ptr<PatchEntity> patch_entity(size_t id) = 0;

    virtual std::shared_ptr<const PatchEntity> patch_entity(size_t id) const = 0;

    virtual std::tuple<std::shared_ptr<PatchEntity>, bool>
            insert(size_t id, std::shared_ptr<PatchEntity> p) = 0;

    virtual size_t erase_patch(size_t id) = 0;

private:
};

template<typename TObject>
class EnablePatchFromThis :
        public PatchEntity,
        public std::enable_shared_from_this<TObject>
{
public:
    /**
    * @name patch
    * @{
    */

    EnablePatchFromThis() { }

    virtual ~EnablePatchFromThis() { }


    virtual TObject &self() = 0;

    virtual TObject const &self() const = 0;

    virtual void refinement() { }

    virtual void coarsen() { }

    virtual std::shared_ptr<PatchEntity>
    patch_entity(size_t id) { return std::dynamic_pointer_cast<PatchEntity>(m_patches_.at(id)); }

    virtual std::shared_ptr<const PatchEntity>
    patch_entity(size_t id) const { return std::dynamic_pointer_cast<const PatchEntity>(m_patches_.at(id)); }


    virtual std::shared_ptr<TObject>
    patch(size_t id) { return (m_patches_.at(id)); }

    virtual std::shared_ptr<const TObject>
    patch(size_t id) const { return (m_patches_.at(id)); }

    virtual size_t erase_patch(size_t id) { return m_patches_.erase(id); };


    virtual std::tuple<std::shared_ptr<PatchEntity>, bool>
    insert(size_t id, std::shared_ptr<PatchEntity> p)
    {
        auto res = insert(id, std::dynamic_pointer_cast<TObject>(p));

        return std::make_tuple(
                std::dynamic_pointer_cast<PatchEntity>(std::get<0>(res)),
                std::get<1>(res));
    };

    virtual std::tuple<std::shared_ptr<TObject>, bool>
    insert(size_t id, std::shared_ptr<TObject> p)
    {
        auto res = m_patches_.insert(std::make_pair(id, p));
        return std::make_tuple(res.first->second, res.second);
    };

    std::map<size_t, std::shared_ptr<TObject>> const &patches() const { return m_patches_; };

private:
    std::map<size_t, std::shared_ptr<TObject>> m_patches_;
};

}} //namespace simpla { namespace mesh

#endif //SIMPLA_PATCH_H
