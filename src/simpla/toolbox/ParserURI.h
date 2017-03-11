//
// Created by salmon on 17-3-11.
//

#ifndef SIMPLA_PARSERURI_H
#define SIMPLA_PARSERURI_H

#include <tuple>
namespace simpla {
namespace toolbox {
/** <scheme , authority ,path ,query,fragment> */
std::tuple<std::string, std::string, std::string, std::string, std::string> ParseURI(std::string const& uri);
}
}
#endif  // SIMPLA_PARSERURI_H
