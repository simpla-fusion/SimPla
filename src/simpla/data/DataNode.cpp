//
// Created by salmon on 17-8-18.
//

#include "DataNode.h"
#include <iomanip>
#include <map>
#include <regex>
#include <vector>
#include "DataEntity.h"
#include "simpla/utilities/ParsingURI.h"

namespace simpla {
namespace data {

DataNode::DataNode() = default;
DataNode::~DataNode() = default;
std::shared_ptr<DataNode> DataNode::New(std::string const& s) {
    if (DataNode::s_num_of_pre_registered_ == 0) { RUNTIME_ERROR << "No database is registered!" << s << std::endl; }
    std::string uri = s.empty() ? "mem://" : s;

    std::string scheme;
    std::string path;
    std::string authority;
    std::string query;
    std::string fragment;

    std::tie(scheme, authority, path, query, fragment) = ParsingURI(uri);
    auto res = Factory<DataNode>::Create(scheme);
    ASSERT(res != nullptr);
    if (SP_SUCCESS != res->Connect(authority, path, query, fragment)) {
        RUNTIME_ERROR << "Fail to connect  Data Backend [ " << scheme << " : " << authority << path << " ]"
                      << std::endl;
    }
    return res;
};

std::shared_ptr<DataEntity> DataNode::GetEntity() const { return DataEntity::New(); }
size_type DataNode::Set(std::string const& uri, std::shared_ptr<DataEntity> const& v) { return 0; }
size_type DataNode::Add(std::string const& uri, std::shared_ptr<DataEntity> const& v) { return 0; }
size_type DataNode::Set(std::shared_ptr<DataNode> const& v) {
    return v == nullptr ? 0 : v->Foreach([&](std::string k, std::shared_ptr<const DataNode> node) {
        size_type count = 0;
        if (node->type() == DataNode::DN_ENTITY) {
            count += Set(k, node->GetEntity());
        } else {
            count += Set(k, node->GetEntity());
        }
        return count;
    });
};
size_type DataNode::Add(std::shared_ptr<DataNode> const& v) { return 0; };
std::istream& DataNode::Parse(std::istream& is) {
    Parse(std::string(std::istreambuf_iterator<char>(is), {}));
    return is;
}
std::ostream& DataNode::Print(std::ostream& os, int indent) const {
    if (this->type() == DataNode::DN_ARRAY) {
        os << "[";
        bool is_first = true;
        bool new_line = this->size() > 1;
        this->Foreach([&](auto k, auto v) {
            if (is_first) {
                is_first = false;
            } else {
                os << ", ";
            }
            if (new_line && v->type() != DataNode::DN_ENTITY) { os << std::endl << std::setw(indent + 1) << " "; }
            v->Print(os, indent + 1);
            return 1;
        });
        os << "]";
    } else if (this->type() == DataNode::DN_TABLE) {
        os << "{ ";
        bool is_first = true;
        bool new_line = this->size() > 1;
        this->Foreach([&](auto k, auto v) {
            if (is_first) {
                is_first = false;
            } else {
                os << ", ";
            }
            if (new_line) { os << std::endl << std::setw(indent + 1) << " "; }
            FancyPrint(os, k, indent);
            os << " = ";
            v->Print(os, indent + 1);
            return 1;
        });

        if (new_line) { os << std::endl << std::setw(indent) << " "; }
        os << "}";

    } else if (this->type() == DataNode::DN_ENTITY) {
        this->GetEntity()->Print(os, indent + 1);
    }
    return os;
}
std::ostream& operator<<(std::ostream& os, DataNode const& entry) { return entry.Print(os, 0); }

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
//std::pair<std::string, std::shared_ptr<DataNode>> RecursiveFindNode(std::shared_ptr<DataNode> root, std::string uri,
//                                                                    int flag) {
//    std::pair<std::string, std::shared_ptr<DataNode>> res{"", root};
//
//    if (uri.empty() || uri == ".") { return res; }
//
//    if (uri[0] == '/') {
//        root = root->Root();
//    } else if (uri.substr(0, 3) == "../") {
//        root = root->Parent();
//        uri = uri.substr(3);
//    }
//    std::smatch uri_match_result;
//
//    if (!std::regex_match(uri, uri_match_result, match_path_regex)) {
//        RUNTIME_ERROR << "illegal URI: [" << uri << "]" << std::endl;
//    }
//    std::string path = uri_match_result.str(2);
//
//    if (!path.empty()) {
//        std::smatch sub_match_result;
//        auto t = root;
//
//        for (auto pos = path.cbegin(), end = path.cend();
//             std::regex_search(pos, end, sub_match_result, sub_group_regex); pos = sub_match_result.suffix().first) {
//            std::string k = sub_match_result.str(1);
//            res.second = t->Get(k);
//            t = res.second;
//            if (res.second == nullptr) {
//                res.first = sub_match_result.suffix().str() + uri_match_result[3].str();
//                break;
//            }
//        }
//    }
//    auto key = uri_match_result.str(3);
//    if (!key.empty()) {
//        res.first = "";
//        res.second = res.second->Get(key);
//    }
//    return res;
//};

size_type DataNode::Set(KeyValue const& kv) { return Set(kv.m_node_); }
size_type DataNode::Add(KeyValue const& kv) { return Add(kv.m_node_); }

DataNode& DataNode::operator=(KeyValue const& v) {
    Set(v.m_node_);
    return *this;
}
DataNode& DataNode::operator=(std::initializer_list<KeyValue> const& u) {
    for (auto const& v : u) { Set(v.m_node_); }
    return *this;
}
//DataNode& DataNode::operator+=(KeyValue const& u) {
//    Add(u.m_node_);
//    return *this;
//}
//DataNode& DataNode::operator+=(std::initializer_list<KeyValue> const& u) {
//    for (auto const& v : u) { Add(v.m_node_); }
//    return *this;
//}

}  // namespace data {
}  // namespace simpla {
