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
#include <simpla/utilities/sp_def.h>
#include <typeindex>
namespace simpla {
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
    SPObject(std::string const &s_name="");
    virtual ~SPObject();

    SP_DEFAULT_CONSTRUCT(SPObject);

    void SetGUID(id_type id);
    id_type GetGUID() const;

    std::string const &GetName() const;
    void SetName(std::string const &);

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
    virtual void SetUp();       //!< repeat invoke,
    virtual void TearDown();    //!< repeat invoke,
    virtual void Finalize();    //!< invoke once, after everything

    design_pattern::Signal<void(SPObject *)> PreInitialize;
    design_pattern::Signal<void(SPObject *)> PostInitialize;
    design_pattern::Signal<void(SPObject *)> PreSetUp;
    design_pattern::Signal<void(SPObject *)> PostSetUp;
    design_pattern::Signal<void(SPObject *)> PreTearDown;
    design_pattern::Signal<void(SPObject *)> PostTearDown;
    design_pattern::Signal<void(SPObject *)> PreFinalize;
    design_pattern::Signal<void(SPObject *)> PostFinalize;
    design_pattern::Signal<void(SPObject *)> PostChanged;

    void DoInitialize();
    void DoFinalize();
    void DoSetUp();
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
}  // namespace simpla { namespace base

#endif  // SIMPLA_OBJECT_H
