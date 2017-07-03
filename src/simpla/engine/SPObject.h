/**
 * @file object.h
 * @author salmon
 * @date 2015-12-16.
 */

#ifndef SIMPLA_OBJECT_H
#define SIMPLA_OBJECT_H

#include <memory>
#include <mutex>
#include <typeinfo>

#include <simpla/SIMPLA_config.h>
#include <simpla/utilities/Signal.h>
#include <typeindex>
#include "DataPack.h"
namespace simpla {
namespace engine {
#define SP_DECLARE_NAME(_CLASS_NAME_)                                \
    virtual std::string GetClassName() const { return ClassName(); } \
    static std::string ClassName() { return __STRING(_CLASS_NAME_); }

#define SP_DEFAULT_CONSTRUCT(_CLASS_NAME_)                 \
    _CLASS_NAME_(this_type const &other) = delete;         \
    _CLASS_NAME_(this_type &&other) = delete;              \
    this_type &operator=(this_type const &other) = delete; \
    this_type &operator=(this_type &&other) = delete;

#define SP_OBJECT_BASE(_BASE_CLASS_NAME_)                                                            \
   private:                                                                                          \
    typedef _BASE_CLASS_NAME_ this_type;                                                             \
                                                                                                     \
   public:                                                                                           \
    virtual bool isA(const std::type_info &info) const { return typeid(_BASE_CLASS_NAME_) == info; } \
    template <typename _UOTHER_>                                                                     \
    bool isA() const {                                                                               \
        return isA(typeid(_UOTHER_));                                                                \
    }                                                                                                \
    template <typename U_>                                                                           \
    U_ &cast_as() {                                                                                  \
        return *dynamic_cast<U_ *>(this);                                                            \
    }                                                                                                \
    template <typename U_>                                                                           \
    U_ const &cast_as() const {                                                                      \
        return *dynamic_cast<U_ const *>(this);                                                      \
    }                                                                                                \
    virtual std::type_info const &GetTypeInfo() const { return typeid(_BASE_CLASS_NAME_); }

/**
 * @brief define the common part of the derived class
 */
#define SP_OBJECT_HEAD(_CLASS_NAME_, _BASE_CLASS_NAME_)                                 \
   public:                                                                              \
    bool isA(std::type_info const &info) const override {                               \
        return typeid(_CLASS_NAME_) == info || _BASE_CLASS_NAME_::isA(info);            \
    }                                                                                   \
    std::type_info const &GetTypeInfo() const override { return typeid(_CLASS_NAME_); } \
                                                                                        \
   private:                                                                             \
    typedef _BASE_CLASS_NAME_ base_type;                                                \
    typedef _CLASS_NAME_ this_type;                                                     \
                                                                                        \
   public:

/**
 * @addtogroup concept
 * @{
 */

/**
 * @brief  define the common part of the base class
 */

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
         state initialized         {
           state prepared         {
                state locked{
                }
            }
            }
        null -->   initialized: Initialize
        initialized --> prepared     : PreProcess
        prepared --> locked    : lock
        locked  -->  prepared  : unlock
        prepared --> initialized     : PostProcess
        initialized --> null     : Finalize
    @enduml


 **/

class SPObject {
    SP_OBJECT_BASE(SPObject)
   public:
    SPObject(std::string const &s_name = "");
    virtual ~SPObject();
    SPObject(SPObject const &other);
    SPObject(SPObject &&other);
    void swap(SPObject &other);
    void SetGUID(id_type id);
    id_type GetGUID() const;

    void SetName(std::string const &s_name);
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

    virtual void Initialize();  //!< invoke once, before everything
    virtual void Update();      //!< repeat invoke, Update object after modified
    virtual void TearDown();    //!< repeat invoke,
    virtual void Finalize();    //!< invoke once, after everything

    design_pattern::Signal<void(SPObject *)> PreInitialize;
    design_pattern::Signal<void(SPObject *)> PostInitialize;
    design_pattern::Signal<void(SPObject *)> PreUpdate;
    design_pattern::Signal<void(SPObject *)> PostUpdate;
    design_pattern::Signal<void(SPObject *)> PreTearDown;
    design_pattern::Signal<void(SPObject *)> PostTearDown;
    design_pattern::Signal<void(SPObject *)> PreFinalize;
    design_pattern::Signal<void(SPObject *)> PostFinalize;
    design_pattern::Signal<void(SPObject *)> PostChanged;

    void DoInitialize();
    void DoFinalize();
    void DoUpdate();
    void DoTearDown();

   private:
    struct pimpl_s;
    std::unique_ptr<pimpl_s> m_pimpl_;
};

// unsigned int state() const { return m_state_; }
//    bool isNull() const { return m_state_ == NULL_STATE; }
//    bool isInitialized() const { return m_state_ >= INITIALIZED; }
//    bool isPrepared() const { return m_state_ >= PREPARED; }
//    bool isLocked() const { return m_state_ == LOCKED; }
//    bool isValid() const { return m_state_ == PREPARED; }
//
//    unsigned int NextState();
//    unsigned int PrevState();
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
//     * @brief Finalize object. This function should be invoked _ONLY ONCE_ before Finalize()
//     * @startuml
//     * title  Finalize()
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
//    virtual void Finalize();
//    bool TryFinalize();
/** @} */
#define NULL_ID static_cast<id_type>(-1)
}  // namespace engine{
}  // namespace simpla { namespace base

#endif  // SIMPLA_OBJECT_H
