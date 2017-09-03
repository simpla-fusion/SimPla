/**
 * @file object.h
 * @author salmon
 * @date 2015-12-16.
 */

#ifndef SIMPLA_OBJECT_H
#define SIMPLA_OBJECT_H
#include "simpla/SIMPLA_config.h"

#include <memory>
#include <mutex>
#include <typeindex>
#include <typeinfo>

#include "simpla/data/DataNode.h"
#include "simpla/utilities/Factory.h"
#include "simpla/utilities/ObjectHead.h"
#include "simpla/utilities/Signal.h"

namespace simpla {
namespace engine {
#define NULL_ID static_cast<id_type>(-1)

/**
 *
 *  @brief every thing is an Object
 *
 *  @details
 *
 *  ## Summary
 *    Life cycle control of an object
 *
 *
 * ## Requirements
 *  Class \c R implementing the concept of @ref Object must define:
 *   Pseudo-Signature                   | Semantics
 *	 -----------------------------------|----------
 * 	 \code R()                 \endcode | Constructor;
 * 	 \code virtual ~R()        \endcode | Destructor
 * 	 \code id_type id()        \endcode |  return id
 *   \code bool isNull()       \endcode |  return m_state_ == NULL_STATE
 *   \code bool isValid()      \endcode |  return m_state_ == VALID
 *   \code bool isReady()      \endcode |  return m_state_ == READY
 *   \code bool isLocked()     \endcode |  return m_state_ == LOCKED
 *   \code void Initialize()   \endcode |  Initial setup. This function should be invoked ONLY ONCE after Deploy()
 *   \code void PreProcess()   \endcode |  This function should be called before operation
 *   \code void Lock()         \endcode |  lock object for concurrent operation
 *   \code bool TryLock()      \endcode |  return true if object is locked
 *   \code void Unlock()       \endcode |  unlock object after concurrent operation
 *   \code void PostProcess()  \endcode |  This function should be called after operation
 *   \code void Finalize()     \endcode |  Finalize object. This function should be invoked ONLY ONCE  before Destroy()
 *
 *
  @startuml
        state null{
        }
        state initialized {
        }
        state modified {
        }
        state ready{
        }

        null -->   initialized: Initialize
        initialized --> modified : Click
        modified --> ready    : Tag/Update
        ready  -->  modified  : Click
        ready --> initialized: TearDown
        modified --> null     : ResetTag/Finalize

    @enduml


 **/

class SPObject : public Factory<SPObject>, public std::enable_shared_from_this<SPObject> {
    typedef Factory<SPObject> base_type;
    typedef SPObject this_type;

   public:
    static std::string GetFancyTypeName_s() { return "SPObject"; }
    virtual std::string GetFancyTypeName() const { return GetFancyTypeName_s(); }
    template <typename U>
    static int RegisterCreator() noexcept {
        return Factory<SPObject>::RegisterCreator<U>(U::GetFancyTypeName_s());
    };
   protected:
    SPObject();

   public:
    ~SPObject() override;


    virtual std::shared_ptr<data::DataNode> Serialize() const;
    virtual void Deserialize(std::shared_ptr<const data::DataNode>);
    static std::shared_ptr<SPObject> New(std::string const &v);
    static std::shared_ptr<SPObject> New(std::shared_ptr<const data::DataNode> tdb);
    static std::shared_ptr<SPObject> NewAndSync(std::shared_ptr<const data::DataNode> v);

    template <typename TOBJ>
    static std::shared_ptr<TOBJ> NewAndSyncT(std::shared_ptr<data::DataNode> const &v) {
        return std::dynamic_pointer_cast<TOBJ>(NewAndSync(v));
    };

    std::shared_ptr<data::DataNode> db() const;
    std::shared_ptr<data::DataNode> db();
    id_type GetGUID() const;
    void SetName(std::string const &);
    std::string const &GetName() const;

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

   private:
    struct pimpl_s;
    pimpl_s *m_pimpl_;
};

std::ostream &operator<<(std::ostream &os, SPObject const &obj);
std::istream &operator>>(std::istream &is, SPObject &obj);

#define SP_OBJECT_PROPERTY(_TYPE_, _NAME_)                                  \
    void Set##_NAME_(_TYPE_ _v_) { db()->SetValue(__STRING(_NAME_), _v_); } \
    _TYPE_ Get##_NAME_() const { return db()->template GetValue<_TYPE_>(__STRING(_NAME_)); }

#define SP_OBJECT_HEAD(_CLASS_NAME_, _BASE_)                                                                        \
   public:                                                                                                          \
    static std::string GetFancyTypeName_s() { return _BASE_::GetFancyTypeName_s() + "." + __STRING(_CLASS_NAME_); } \
    virtual std::string GetFancyTypeName() const override { return GetFancyTypeName_s(); }                          \
    static bool _is_registered;                                                                                     \
                                                                                                                    \
   private:                                                                                                         \
    typedef _BASE_ base_type;                                                                                       \
    typedef _CLASS_NAME_ this_type;                                                                                 \
                                                                                                                    \
   public:                                                                                                          \
   protected:                                                                                                       \
    _CLASS_NAME_();                                                                                                 \
                                                                                                                    \
   public:                                                                                                          \
    ~_CLASS_NAME_() override;                                                                                       \
                                                                                                                    \
    std::shared_ptr<simpla::data::DataNode> Serialize() const override;                                             \
    void Deserialize(std::shared_ptr<const simpla::data::DataNode> cfg) override;                                   \
                                                                                                                    \
   private:                                                                                                         \
    struct pimpl_s;                                                                                                 \
    pimpl_s *m_pimpl_ = nullptr;                                                                                    \
                                                                                                                    \
    template <typename U, typename... Args>                                                                         \
    static std::shared_ptr<U> _TryCreate(std::integral_constant<bool, false> _, Args &&... args) {                  \
        return std::shared_ptr<U>(new U(std::forward<Args>(args)...));                                              \
    }                                                                                                               \
    template <typename U, typename... Args>                                                                         \
    static std::shared_ptr<U> _TryCreate(std::integral_constant<bool, true> _, Args &&... args) {                   \
        return std::dynamic_pointer_cast<_CLASS_NAME_>(base_type::New(std::forward<Args>(args)...));                \
    }                                                                                                               \
                                                                                                                    \
   public:                                                                                                          \
    template <typename... Args>                                                                                     \
    static std::shared_ptr<_CLASS_NAME_> New(Args &&... args) {                                                     \
        return _TryCreate<_CLASS_NAME_>(typename std::is_abstract<_CLASS_NAME_>(), std::forward<Args>(args)...);    \
    };                                                                                                              \
    static std::shared_ptr<_CLASS_NAME_> New(std::shared_ptr<const data::DataNode> node) {                          \
        auto res = SPObject::New(__STRING(_CLASS_NAME_) + std::string(".") +                                        \
                                 node->GetValue<std::string>("_TYPE_", "default"));                                 \
        res->Deserialize(node);                                                                                     \
        return std::dynamic_pointer_cast<_CLASS_NAME_>(res);                                                        \
    };

#define SP_OBJECT_REGISTER(_CLASS_NAME_) \
    bool _CLASS_NAME_::_is_registered = simpla::engine::SPObject::RegisterCreator<_CLASS_NAME_>();

//    static std::shared_ptr<_CLASS_NAME_> New(std::shared_ptr<const data::DataNode> v) {                                \
//        auto s_type = v->as<std::string>("");                                                                          \
//        if (s_type.empty() && std::dynamic_pointer_cast<const data::DataTable>(v) != nullptr) {                        \
//            s_type = std::dynamic_pointer_cast<const data::DataTable>(v)->GetEntity<std::string>("Type", "");           \
//        }                                                                                                              \
//        auto res = base_type::Create(std::string(TagName()) + (s_type.empty() ? "" : ".") + s_type);                   \
//        res->Deserialize(v);                                                                                           \
//        return std::dynamic_pointer_cast<_CLASS_NAME_>(res);                                                           \
//    };

//
//    /**
//     * @brief Initial setup.
//       @startuml
//          title  TryInitialize()
//          (*) --> if "isInitialized()?" then
//                      --> [true] (*)
//                  else
//                      --> [false] "state = INITIALIZED"
//                      --> (*)
//
//                 endif
//      @enduml
//     */
//    bool TryInitialize();
//    /**
//     *  @brief Initial setup.
//     */
//    virtual void InitializeConditionPatch();
//
//    /**
//     * @brief Initial setup. This function should be invoked _ONLY ONCE_  after InitializeConditionPatch()
//     * @startuml
//     *    title  TryPreProcess()
//     *    (*) --> if "isPrepared()?" then
//     *                --> [true] (*)
//     *            else
//     *                --> [false] InitializeConditionPatch()
//     *                --> "state = PREPARED"
//     *                --> (*)
//     *           endif
//     * @enduml
//    */
//    bool TryPreProcess();
//    virtual void PreProcess();  //< This function should be called before operation
//
//    /**
//     * @startuml
//     *    title  lock()
//     *    (*) -down-> START
//     *            if "isLocked()?" then
//     *                --> [true] wait()
//     *                -up-> START
//     *            else
//     *                 -left-> [false] " state=LOCKED"
//     *                 --> (*)
//     *           endif
//     * @enduml
//     */
//    virtual void Lock();
//
//    /**
//    * @startuml
//    *    title  TryLock()
//    *    (*) --> check
//     *           if "isLocked()?" then
//    *                -left->   [true] check
//    *            else
//     *                --> PreProcess()
//    *                 --> [false] " state=LOCKED"
//    *                 --> (*)
//    *           endif
//    * @enduml
//    */
//    virtual bool TryLock();
//
//    /**
//     * @startuml
//     *    title  Unlock()
//     *    (*) --> if "isLocked()?" then
//     *                --> [true] "--state"
//     *                --> (*)
//     *            else
//     *                --> [false]   (*)
//     *           endif
//     * @enduml
//     */
//    virtual void Unlock();
//
//    /**
//     * @brief   This function should be called after operation
//     * @startuml
//     *    title  PostProcess()
//     *    (*) --> if "isPrepared()?" then
//     *                --> [true]  Unlock()
//     *                --> "--state "
//     *                --> (*)
//     *            else
//     *                --> [false]   (*)
//     *           endif
//     * @enduml
//     */
//    bool TryPostProcess();
//    virtual void PostProcess();
//
//    /**
//     * @brief Finalize object. This function should be invoked _ONLY ONCE_ before DoFinalize()
//     * @startuml
//     * title  DoFinalize()
//     * (*) --> if "isInitialized()?" then
//     *     -->[true] PostProcess()
//     *     --> "--state "
//     *     --> (*)
//     * else
//     *     -->[false] (*)
//     * endif
//     * @enduml
//     *
//     *
//     */
//
//    virtual void DoFinalize();
//    bool TryFinalize();
/** @} */
}  // namespace engine{
}  // namespace simpla { namespace base

#endif  // SIMPLA_OBJECT_H
