//
// Created by salmon on 17-3-6.
//
#include "DataBackendFactroy.h"
#include "DataBackend.h"
#include "DataBackendLua.h"
#include "DataBackendMemory.h"
namespace simpla {
namespace data {
std::shared_ptr<DataBackend> CreateDataBackendFromFile(std::string const &url, std::string const &status) {
    if (url == "") { return std::make_shared<DataBackendMemory>(); }

    std::shared_ptr<DataBackend> res = nullptr;
    std::string ext = "";
    size_type pos = url.find_last_of('.');
    if (pos != std::string::npos) { ext = url.substr(pos + 1); }
    if (ext == "lua") {
        res = std::make_shared<DataBackendLua>(url, status);
    } else {
        res = std::make_shared<DataBackendMemory>(url, status);
    }
    return res;
};
}  // namespace data{
}  // namespace simpla{