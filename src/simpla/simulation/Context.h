/**
 * @file context.h
 *
 * @date    2014-9-18  AM9:33:53
 * @author salmon
 */

#ifndef CORE_APPLICATION_CONTEXT_H_
#define CORE_APPLICATION_CONTEXT_H_

#include <simpla/SIMPLA_config.h>

#include <memory>
#include <list>
#include <map>
#include <simpla/toolbox/Log.h>

#include <simpla/concept/Object.h>
#include <simpla/concept/Printable.h>
#include <simpla/concept/Serializable.h>


#include <simpla/mesh/EntityIdRange.h>
#include <simpla/mesh/Mesh.h>
#include <simpla/mesh/DataBlock.h>
#include <simpla/manifold/Atlas.h>
#include <simpla/mesh/TransitionMap.h>
#include <simpla/mesh/DomainBase.h>
#include <simpla/toolbox/IOStream.h>
#include <simpla/data/DataEntityTable.h>

namespace simpla { namespace simulation
{

/**
 *  life cycle of a simpla::Context
 *
 *
 *
 * constructure->initialize -> load -> +-[ add_*    ]  -> deploy-> +- [ next_step   ] -> save -> [ teardown ] -> destructure
 *                                     |-[ register*]              |- [ check_point ]
 *                                                                 |- [ print       ]
 *
 *
 *
 *
 *
 *
 */
class Context :
        public Object,
        public concept::Printable,
        public concept::Serializable
{

public:
    SP_OBJECT_HEAD(Context, Object)const std::type_index &;

    Context() : Object() {};

    virtual ~Context() {};

    virtual std::ostream &print(std::ostream &os, int indent) const { return os; };

    virtual void load(data::DataEntityTable const &) { UNIMPLEMENTED; }

    virtual void save(data::DataEntityTable *) const { UNIMPLEMENTED; }

    virtual void initialize(int argc = 0, char **argv = nullptr)=0;

    virtual void deploy() {};

    virtual void teardown() {};

    virtual bool is_valid() const { return true; };

    virtual toolbox::IOStream &check_point(toolbox::IOStream &os) const { return os; };

    virtual size_type step() const =0;

    virtual Real time() const =0;

    virtual void next_time_step(Real dt)=0;


};


}}// namespace simpla{namespace simulation


#endif /* CORE_APPLICATION_CONTEXT_H_ */
