//
// Created by salmon on 17-3-6.
//
#include "DataTableFactroy.h"
#include "DataBackend.h"
#include "DataBackendLua.h"
#include "DataBackendMemory.h"
namespace simpla {
namespace data {
std::shared_ptr<DataBackend> CreateDataBackendFromFile(std::string const &url) {
    std::shared_ptr<DataBackend> res(nullptr);
    std::string ext = "";
    size_type pos = url.find_last_of('.');
    if (pos != std::string::npos) { ext = url.substr(pos + 1); }
    if (ext == "lua") {
        res = std::dynamic_pointer_cast<DataBackend>(std::make_shared<DataBackendLua>(url));
    } else {
        res = std::dynamic_pointer_cast<DataBackend>(std::make_shared<DataBackendMemory>(url));
    }
    return res;
};
}  // namespace data{
}  // namespace simpla{