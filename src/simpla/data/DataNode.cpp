//
// Created by salmon on 17-8-18.
//

#include "DataNode.h"
#include <iomanip>
#include <map>
#include <vector>
#include "DataBase.h"
#include "DataEntity.h"
namespace simpla {
namespace data {

DataNode::DataNode() = default;
DataNode::~DataNode() = default;
std::shared_ptr<DataNode> DataNode::New(std::string const& s) { return data::DataBase::New(s)->Root(); }

std::ostream& Print(std::ostream& os, std::shared_ptr<const DataNode> const& entry, int indent) {
    if (entry->isArray()) {
        os << "[ ";
        auto it = entry->FirstChild();
        Print(os, it, indent + 1);
        it = it->Next();
        while (it != nullptr) {
            os << " , ";
            Print(os, it, indent);
        }
        os << " ]";
    } else if (entry->isTable()) {
        os << "{ ";
        auto it = entry->FirstChild();
        if (it != nullptr) {
            os << std::endl << std::setw(indent) << "\"" << it->GetKey() << "\" = ";
            std::cout << it->GetKey() << std::endl;
            Print(os, it, indent + 1);
            it = it->Next();
            while (it != nullptr) {
                os << "," << std::endl << std::setw(indent) << "\"" << it->GetKey() << "\" = ";
                Print(os, it, indent + 1);
                it = it->Next();
            }
            if (entry->GetNumberOfChildren() > 1) { os << std::endl << std::setw(indent) << " "; }
            os << "}";
        }
    } else if (entry->isEntity()) {
        os << *entry->GetEntity();
    }
    return os;
}
std::ostream& operator<<(std::ostream& os, DataNode const& entry) { return Print(os, entry.shared_from_this(), 0); }

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
std::pair<std::shared_ptr<DataNode>, std::string> RecursiveFindNode(std::shared_ptr<DataNode> const& root,
                                                                    std::string const& uri, int flag) {
    std::pair<std::shared_ptr<DataNode>, std::string> res{root, ""};

    if (uri.empty()) { return res; }
    std::smatch uri_match_result;

    if (!std::regex_match(uri, uri_match_result, match_path_regex)) {
        RUNTIME_ERROR << "illegal URI: [" << uri << "]" << std::endl;
    }
    std::string path = uri_match_result.str(2);

    if (!path.empty()) {
        std::smatch sub_match_result;
        auto t = root;

        for (auto pos = path.cbegin(), end = path.cend();
             std::regex_search(pos, end, sub_match_result, sub_group_regex); pos = sub_match_result.suffix().first) {
            std::string k = sub_match_result.str(1);

            res.first = t->GetNode(k, flag & (~DataNode::RECURSIVE));
            //        try
            t = res.first;
            if (res.first == nullptr) {
                res.second = sub_match_result.suffix().str() + uri_match_result[3].str();
                break;
            }
        }
    }
    auto key = uri_match_result.str(3);
    if (!key.empty()) {
        res.first = res.first->GetNode(key, flag & (~DataNode::RECURSIVE));
        res.second = "";
    }
    return res;
};

}  // namespace data {
}  // namespace simpla {
