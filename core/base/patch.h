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
#include <tuple>
#include "../gtl/ntuple.h"
#include "../gtl/utilities/log.h"
#include "../gtl/primitives.h"
#include "attribute.h"

namespace simpla { namespace mesh
{
template<typename ...> class Patch;

template<typename ...> class Layout;


class PatchEntity : public AttributeBase
{
public:
    PatchEntity() { };

    virtual ~PatchEntity() { }

    virtual void refinement() = 0;

    virtual void coarsen() = 0;

    virtual void sync_patches() { };


    virtual std::shared_ptr<PatchEntity>
    patch_entity(size_t id) { return (m_patches_.at(id)); }

    virtual std::shared_ptr<const PatchEntity>
    patch_entity(size_t id) const { return (m_patches_.at(id)); }


    virtual std::tuple<std::shared_ptr<PatchEntity>, bool>
    insert(size_t id, std::shared_ptr<PatchEntity> p)
    {
        auto res = m_patches_.insert(std::make_pair(id, p));
        return std::make_tuple(res.first->second, res.second);
    };

    virtual size_t erase_patch(size_t id) { return m_patches_.erase(id); };

    virtual std::tuple<nTuple<Real, 3>, nTuple<Real, 3>> get_box() const
    {
        return std::make_tuple(nTuple<Real, 3> {0, 0, 0}, nTuple<Real, 3> {1, 1, 1});
    };

    virtual nTuple<size_t, 3> get_dimensions() const
    {
        return nTuple<size_t, 3>{1, 1, 1};
    };

    std::map<size_t, std::shared_ptr<PatchEntity>> const &patches() const { return m_patches_; };

private:
    std::map<size_t, std::shared_ptr<PatchEntity>> m_patches_;
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


    virtual std::shared_ptr<TObject>
    patch(size_t id) { return (std::dynamic_pointer_cast<TObject>(patch_entity(id))); }

    virtual std::shared_ptr<const TObject>
    patch(size_t id) const { return (std::dynamic_pointer_cast<const TObject>(patch_entity(id))); }


    virtual std::tuple<std::shared_ptr<TObject>, bool>
    insert(size_t id, std::shared_ptr<TObject> p)
    {
        auto res = PatchEntity::insert(id, std::dynamic_pointer_cast<PatchEntity>(p));
        return std::make_tuple(std::dynamic_pointer_cast<TObject>(std::get<0>(res)), std::get<1>(res));
    };


};

}} //namespace simpla { namespace mesh

#endif //SIMPLA_PATCH_H
