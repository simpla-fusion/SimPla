//
// Created by salmon on 16-11-29.
//

#ifndef SIMPLA_DEPLOYABLE_H
#define SIMPLA_DEPLOYABLE_H

#include <simpla/SIMPLA_config.h>
#include <simpla/toolbox/Log.h>

namespace simpla { namespace concept
{/**  @ingroup concept   */
/**
 * @brief a type whose life cycle is explicit controlled
 * @details
 * ## Summary
 * Requirements for a type whose instances share ownership between multiple objects;
 *
 * ## Requirements
 *  Class \c R implementing the concept of @ref Configurable must define:
 *   Pseudo-Signature                                      | Semantics
 *	 ------------------------------------------------------|----------
 * 	 \code   R()                                  \endcode | constructor;
 * 	 \code  virtual ~R()                          \endcode | Destructor
 * 	 \code bool is_deployed()                     \endcode | return true after @ref deploy is evaluated, return false after @ref destroy is evaluated
 *   \code bool is_valid()                        \endcode | return true after @ref PreProcess is evaluated, return false after @ref PostProcess is evaluated
 * 	 \code void deploy()                          \endcode | if is_deployed==true then throw RUNTIME_ERROR else set is_deployed =true
 *   \code void PreProcess()                     \endcode |  set is_valid =true
 *   \code void PostProcess()                    \endcode |  set is_valid =false
 *   \code void destroy()                         \endcode |  set is_deployed =false
 */
struct LifeControllable
{

    LifeControllable() : m_is_deployed_(false), m_is_valid_(false) {}

    virtual ~LifeControllable() { Destroy(); }

    virtual bool isDeployed() const { return m_is_deployed_; }

    virtual bool isValid() const { return m_is_valid_; }

    /**
     * @name Life Cycle
     * @{
     */

    virtual void Deploy()
    {
        if (isDeployed()) { RUNTIME_ERROR << "Repeat deploying!" << std::endl; }
        m_is_deployed_ = true;
    };


    virtual void PreProcess() { m_is_valid_ = true; /*add sth here*/}

    virtual void PostProcess() {  /*add sth here*/ m_is_valid_ = false; }

    virtual void Destroy() { m_is_deployed_ = false; };


    /** @}*/

private:
    bool m_is_deployed_ = false;
    bool m_is_valid_ = false;
};
}}//namespace simpla{namespace concept{
#endif //SIMPLA_DEPLOYABLE_H
