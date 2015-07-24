/**
 * @file use_case.cpp
 *
 *  Created on: 2014-11-21
 *      Author: salmon
 */

#include "use_case.h"

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include "../gtl/utilities.h"

namespace simpla {

std::string UseCaseList::add(std::string const &name,
                             std::shared_ptr<UseCase> const &p)
{
    list_[name] = p;
    return "UseCase" + value_to_string(list_.size()) + "_" + name;
}

std::ostream &UseCaseList::print(std::ostream &os)
{
    for (auto const &item : list_)
    {
        os << item.first << std::endl;
    }
    return os;
}

void UseCaseList::run(int argc, char **argv)
{
    for (auto const &item : list_)
    {

        LOGGER << "Case [" << item.first << "] initialize ." << std::endl;

        item.second->init(argc, argv);

        LOGGER << "Case [" << item.first << "] start." << std::endl;

        item.second->body();

        LOGGER << "Case [" << item.first << "] done." << std::endl;
    }
}
}  // namespace simpla
