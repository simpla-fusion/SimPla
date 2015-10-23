/**
 * @file  task.h
 *
 *  Created on: 2015-1-4
 *      Author: salmon
 */

#ifndef CORE_APPLICATION_TASK_H_
#define CORE_APPLICATION_TASK_H_

#include "../gtl/type_traits.h"

namespace simpla
{
class TaskBase
{
    TaskBase(TaskBase &, tags::split);

    virtual ~TaskBase() { };

    virtual void split(TaskBase &) = 0;

    virtual void execute() = 0;

    virtual void setup() = 0;

    virtual void teardown() = 0;

};

template<typename ...>
struct Task;

}  // namespace simpla

#endif /* CORE_APPLICATION_TASK_H_ */
