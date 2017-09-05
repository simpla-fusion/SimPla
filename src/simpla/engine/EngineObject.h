//
// Created by salmon on 17-9-3.
//

#ifndef SIMPLA_ENGOBJECT_H
#define SIMPLA_ENGOBJECT_H

#include "simpla/data/SPObject.h"
#include "simpla/utilities/Signal.h"

namespace simpla {
namespace engine {
class EngineObject : public SPObject {
   public:
    EngineObject();
    ~EngineObject() override;
    void lock();
    void unlock();
    bool try_lock();
    void Tag();
    void Click();
    void ResetTag();
    size_type GetTagCount() const;
    size_type GetClickCount() const;
    bool isModified() const;
    bool isInitialized() const;

    virtual void DoInitialize();  //!< invoke once, before everything
    virtual void DoUpdate();      //!< repeat invoke, Update object after modified
    virtual void DoTearDown();    //!< repeat invoke,
    virtual void DoFinalize();    //!< invoke once, after everything

    design_pattern::Signal<void(SPObject *)> PreInitialize;
    design_pattern::Signal<void(SPObject *)> PostInitialize;
    design_pattern::Signal<void(SPObject *)> PreUpdate;
    design_pattern::Signal<void(SPObject *)> PostUpdate;
    design_pattern::Signal<void(SPObject *)> PreTearDown;
    design_pattern::Signal<void(SPObject *)> PostTearDown;
    design_pattern::Signal<void(SPObject *)> PreFinalize;
    design_pattern::Signal<void(SPObject *)> PostFinalize;

    void Initialize();
    void Finalize();
    void Update();
    void TearDown();

    virtual int Push(std::shared_ptr<data::DataNode> const &);
    virtual std::shared_ptr<data::DataNode> Pop();

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_;
};
}  // namespace engine
}
#endif  // SIMPLA_ENGOBJECT_H
