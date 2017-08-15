//
// Created by salmon on 17-3-9.
//

#ifndef SIMPLA_DATAUTILITY_H
#define SIMPLA_DATAUTILITY_H

#include <iomanip>
#include <regex>
#include <string>
#include "simpla/data/Serializable.h"
#include "simpla/utilities/Factory.h"
#include "simpla/utilities/Log.h"
#include "simpla/utilities/type_traits.h"
namespace simpla {
namespace data {
class DataTable;
class DataEntity;
std::shared_ptr<DataTable> ParseCommandLine(int argc, char **argv);

template <typename U>
std::shared_ptr<DataTable> const &Pack(U const &u, ENABLE_IF((std::is_base_of<Serializable, U>::value))) {
    return u.Pack();
}
template <typename U>
std::shared_ptr<U> Unpack(std::shared_ptr<DataTable> const &d, ENABLE_IF((std::is_base_of<Factory<U>, U>::value))) {
    return U::Create(d);
}
void Pack(std::shared_ptr<DataEntity> const &d, std::ostream &os, std::string const &type, int indent = 0);
std::string AutoIncreaseFileName(std::string filename, std::string const &ext_str);

static std::regex const sub_group_regex(R"(([^/?#]+)/)", std::regex::optimize);
static std::regex const match_path_regex(R"(^(/?([/\S]+/)*)?([^/]+)?$)", std::regex::optimize);

/**
 * @brief Traverse  a hierarchical table base on URI  example: /ab/c/d/e
 *           if ''' return_if_not_exist ''' return when sub table does not exist
 *           else create a new table
 * @tparam T  table type
 * @tparam FunCheckTable
 * @tparam FunGetTable
 * @tparam FunAddTable
 * @param self
 * @param uri
 * @param check check  if sub obj is a table
 * @param get  return sub table
 * @param add  create a new table
 * @param return_if_not_exist
 * @return
 */
template <typename T, typename FunCheckTable, typename FunGetTable, typename FunAddTable>
std::pair<T, std::string> HierarchicalTableForeach(T self, std::string const &uri, FunCheckTable const &check,
                                                   FunGetTable const &get, FunAddTable const &add,
                                                   bool return_if_not_exist = false) {
    if (uri == "") { return std::make_pair(self, std::string("")); }
    std::smatch uri_match_result;

    if (!std::regex_match(uri, uri_match_result, match_path_regex)) {
        RUNTIME_ERROR << "illegal URI: [" << uri << "]" << std::endl;
    }
    std::string path = uri_match_result.str(2);

    if (path == "") { return std::make_pair(self, uri); }

    std::smatch sub_match_result;
    T t = self;

    for (auto pos = path.cbegin(), end = path.cend(); std::regex_search(pos, end, sub_match_result, sub_group_regex);
         pos = sub_match_result.suffix().first) {
        std::string k = sub_match_result.str(1);

        //        try
        {
            if (check(t, k)) {
                t = get(t, k);
            } else if (!return_if_not_exist) {
                t = add(t, k);
            } else {
                return std::make_pair(t, sub_match_result.suffix().str() + uri_match_result[3].str());
            }
        }
        //        catch (...) {
        //            WARNING << std::endl
        //                    << std::setw(25) << std::right << "illegal path [/" << uri << "]" << std::endl
        //                    << std::setw(25) << std::right << "     at here   " << std::setw(&(*pos) -
        //                    &(*uri.cbegin())) << " "
        //                    << " ^" << std::endl;
        //        }
    }

    return std::make_pair(t, uri_match_result[3].str());
};

}  // namespace data{
}  // namespace simpla{
#endif  // SIMPLA_DATAUTILITY_H
