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
    virtual void DoSetUp();   //!< invoke after Object all configure opeation , Set/Deserialize, Disable Set/Deserialize
    virtual void DoUpdate();  //!< repeat invoke, Update object after modified
    virtual void DoTearDown();  //!< repeat invoke, enable Set/Deserialize
    virtual void DoFinalize();  //!< invoke once, after everything

    void Push(std::shared_ptr<data::DataNode> const &) override;
    std::shared_ptr<data::DataNode> Pop() override;

    design_pattern::Signal<void(EngineObject *)> PreSetUp;
    design_pattern::Signal<void(EngineObject *)> PostSetUp;
    design_pattern::Signal<void(EngineObject *)> PreUpdate;
    design_pattern::Signal<void(EngineObject *)> PostUpdate;
    design_pattern::Signal<void(EngineObject *)> PreTearDown;
    design_pattern::Signal<void(EngineObject *)> PostTearDown;

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
        ASSERT(isInitialized());                                  \
        return db()->template GetValue<_TYPE_>(__STRING(_NAME_)); \
    }
}  // namespace engine
}
#endif  // SIMPLA_ENGOBJECT_H
