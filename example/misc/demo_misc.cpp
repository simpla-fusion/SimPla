/**
 * @file demo_misc.cpp
 *
 * @date 2015年4月15日
 * @author salmon
 */

#include "../../core/application/application.h"
#include "../../core/application/use_case.h"
#include "../../core/utilities/utilities.h"
#include "../../core/io/io.h"

using namespace simpla;

USE_CASE(misc," Misc. utilities ")
{
	CHECK(options["Foo"](1.0,3,4,5,6).as<double>());
	CHECK(options["Foo"](2.0).as<double>());
}
