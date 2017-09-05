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
    SP_OBJECT_HEAD(EngineObject, SPObject)
   public:
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
    bool isSetUp() const;
    virtual void DoInitialize();  //!< invoke once, before everything,
    virtual void DoSetUp();  //!< invoke after Object all configure opeation , Set/Deserialize, Disable Set/Deserialize
    virtual int Push(std::shared_ptr<data::DataNode> const &);
    virtual void DoUpdate();  //!< repeat invoke, Update object after modified
    virtual std::shared_ptr<data::DataNode> Pop();
    virtual void DoTearDown();  //!< repeat invoke, enable Set/Deserialize
    virtual void DoFinalize();  //!< invoke once, after everything

    design_pattern::Signal<void(SPObject *)> PreInitialize;
    design_pattern::Signal<void(SPObject *)> PostInitialize;
    design_pattern::Signal<void(SPObject *)> PreSetUp;
    design_pattern::Signal<void(SPObject *)> PostSetUp;
    design_pattern::Signal<void(SPObject *)> PreUpdate;
    design_pattern::Signal<void(SPObject *)> PostUpdate;
    design_pattern::Signal<void(SPObject *)> PreTearDown;
    design_pattern::Signal<void(SPObject *)> PostTearDown;
    design_pattern::Signal<void(SPObject *)> PreFinalize;
    design_pattern::Signal<void(SPObject *)> PostFinalize;

    void Initialize();
    void SetUp();

    void Update();

    void TearDown();
    void Finalize();
};

#define SP_OBJECT_PROPERTY(_TYPE_, _NAME_)                        \
    void Set##_NAME_(_TYPE_ const &_v_) {                         \
        ASSERT(!isSetUp());                                       \
        db()->SetValue(__STRING(_NAME_), _v_);                    \
    }                                                             \
    _TYPE_ Get##_NAME_() const {                                  \
        ASSERT(db() != nullptr);                                  \
        return db()->template GetValue<_TYPE_>(__STRING(_NAME_)); \
    }
}  // namespace engine
}
#endif  // SIMPLA_ENGOBJECT_H
