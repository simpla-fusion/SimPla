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
DataBackend *DataBackendFactory::Create(std::string const &scheme) {
    LOGGER << "Create  [ DataBackend: " << scheme << "]" << std::endl;
    return base_type::Create(scheme);
}
std::vector<std::string> DataBackendFactory::RegisteredBackend() const {
    std::vector<std::string> res;
    for (auto const &item : *this) { res.push_back(item.first); }
    return std::move(res);
};

std::shared_ptr<DataBackend> DataBackend::Create(std::string const &scheme) {
    std::shared_ptr<DataBackend> res(GLOBAL_DATA_BACKEND_FACTORY.Create(scheme));
    if (res == nullptr) { res = std::make_shared<DataBackendMemory>(); }
    return res;
}

}  // namespace data {
}  // namespace simpla {