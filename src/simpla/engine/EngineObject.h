//
// Created by salmon on 17-9-3.
//

#ifndef SIMPLA_ENGOBJECT_H
#define SIMPLA_ENGOBJECT_H

#include "Patch.h"
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
    virtual bool isModified() const;
    virtual bool isInitialized() const;
    virtual bool isSetUp() const;
    virtual void DoInitialize();  //!< invoke once, before everything,
    virtual void DoSetUp();   //!< invoke after Object all configure opeation , Set/Deserialize, Disable Set/Deserialize
    virtual void DoUpdate();  //!< repeat invoke, Update object after modified
    virtual void DoTearDown();  //!< repeat invoke, enable Set/Deserialize
    virtual void DoFinalize();  //!< invoke once, after everything

    //    virtual void Push(const std::shared_ptr<data::DataNode> &);
    //    virtual std::shared_ptr<data::DataNode> Pop() const;

    virtual void Push(const std::shared_ptr<Patch> &);
    virtual std::shared_ptr<Patch> Pop() const;

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

#define SP_OBJECT_PROPERTY(_TYPE_, _NAME_)                             \
   private:                                                            \
    _TYPE_ m_##_NAME_##_;                                              \
                                                                       \
   public:                                                             \
    void Set##_NAME_(_TYPE_ const &_v_) {                              \
        if (this->isSetUp()) {                                         \
            WARNING << "Object is set up, can not change properties."; \
        } else {                                                       \
            m_##_NAME_##_ = _v_;                                       \
            this->db()->SetValue(__STRING(_NAME_), _v_);               \
        }                                                              \
    }                                                                  \
    _TYPE_ Get##_NAME_() const { return m_##_NAME_##_; }
}  // namespace engine
}
#endif  // SIMPLA_ENGOBJECT_H
