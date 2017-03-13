//
// Created by salmon on 17-3-9.
//

#include "DataBackend.h"
#include <iomanip>
#include <string>

#include <simpla/toolbox/ParserURI.h>
#include <regex>
#include "DataBackendMemory.h"
#include "DataEntity.h"
#include "DataTable.h"
namespace simpla {
namespace data {
DataBackendFactory::DataBackendFactory() : base_type() { RegisterDefault(); };
DataBackendFactory::~DataBackendFactory(){};
std::shared_ptr<DataBackend> DataBackendFactory::Create(std::string const &scheme) {
    LOGGER << "CreateNew  [ DataBackend: " << scheme << "]" << std::endl;
    std::shared_ptr<DataBackend> res{base_type::Create(scheme)};
    if (res == nullptr) { res = std::make_shared<DataBackendMemory>(); }
    return res;
}
std::vector<std::string> DataBackendFactory::RegisteredBackend() const {
    std::vector<std::string> res;
    for (auto const &item : *this) { res.push_back(item.first); }
    return std::move(res);
};

namespace detail {
static std::regex uri_regex(R"(^(/(([^/?#:]+)/)*)*([^/?#:]*)$)", std::regex::extended | std::regex::optimize);
static std::regex sub_dir_regex(R"(([^/?#:]+)/)", std::regex::extended | std::regex::optimize);
}

}  // namespace data {
}  // namespace simpla {