//
// Created by salmon on 17-3-9.
//

#include "DataBase.h"
#include <iomanip>
#include <string>

#include "DataTable.h"
//#include "db/DataBaseHDF5.h"
//#include "db/DataBaseLua.h"
//#include "db/DataBaseMemory.h"
//#include "db/DataBaseStdIO.h"
#include "simpla/utilities/Factory.h"
#include "simpla/utilities/ParsingURI.h"
namespace simpla {
namespace data {
// int DataBase::s_num_of_pre_registered_ = DataBaseMemory::_is_registered +  //
//                                         DataBaseHDF5::_is_registered +    //
//                                         DataBaseLua::_is_registered +     //
//                                         DataBaseStdIO::_is_registered;
std::shared_ptr<DataBase> DataBase::New(std::string const& uri) {
    if (uri.empty()) { return nullptr; }
//    ASSERT(data::DataBase::s_num_of_pre_registered_ > 0);

    std::string scheme;
    std::string path;
    std::string authority;
    std::string query;
    std::string fragment;

    std::tie(scheme, authority, path, query, fragment) = ParsingURI(uri);
    auto res = Factory<DataBase>::Create(scheme);
    ASSERT(res != nullptr);
    if (SP_SUCCESS == res->Connect(authority, path, query, fragment)) {
        VERBOSE << "Connect  Data Backend [ " << scheme << " : " << authority << path << " ]" << std::endl;
    } else {
        RUNTIME_ERROR << "Fail to connect  Data Backend [ " << scheme << " : " << authority << path << " ]"
                      << std::endl;
    }

    return res;
};

}  // namespace data {
}  // namespace simpla {