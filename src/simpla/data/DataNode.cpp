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
int DataNode::s_num_of_pre_registered_ = 0;

DataNode::DataNode(eNodeType etype) : m_type_(etype) {}
DataNode::~DataNode() {
    if (isRoot()) { Flush(); }
};
std::shared_ptr<DataNode> DataNode::New(std::string const& s) {
    //    if (DataNode::s_num_of_pre_registered_ == 0) { RUNTIME_ERROR << "No database is registered!" << s <<
    //    std::endl; }
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
std::shared_ptr<DataNode> DataNode::New(eNodeType e_type, std::string const& url) {
    return New(url)->CreateNode(e_type);
}

KeyValue::KeyValue(std::string k) : m_key_(std::move(k)), m_node_(DataNode::New(DataLight::New(true))) {}
KeyValue::KeyValue(KeyValue const& other) = default;
KeyValue::KeyValue(KeyValue&& other) noexcept = default;
KeyValue::~KeyValue() = default;
KeyValue& KeyValue::operator=(KeyValue const& other) {
    m_node_ = DataNode::New(DataNode::DN_TABLE, "");
    m_node_->Set(other.m_key_, other.m_node_);
    return *this;
}

template <typename U>
std::shared_ptr<DataNode> make_node(U const& u) {
    return DataNode::New(DataLight::New(u));
}
std::shared_ptr<DataNode> make_node(KeyValue const& kv) {
    auto res = DataNode::New(DataNode::DN_TABLE);
    res->Set(kv.m_key_, kv.m_node_);
    return res;
}
std::shared_ptr<DataNode> make_node(std::initializer_list<KeyValue> const& u) {
    auto res = DataNode::New(DataNode::DN_TABLE);
    for (auto const& v : u) { res->Set(v.m_key_, v.m_node_); }
    return res;
}
template <typename U>
std::shared_ptr<DataNode> make_node(std::initializer_list<U> const& u, ENABLE_IF(traits::is_light_data<U>::value)) {
    return DataNode::New(DataLight::New(u));
}
template <typename U>
std::shared_ptr<DataNode> make_node(std::initializer_list<U> const& u, ENABLE_IF(!traits::is_light_data<U>::value)) {
    auto res = DataNode::New()->CreateNode(DataNode::DN_ARRAY);
    for (auto const& v : u) { res->Add(make_node(v)); }
    return res;
}
KeyValue& KeyValue::operator=(std::initializer_list<KeyValue> const& u) {
    m_node_ = make_node(u);
    return *this;
}
KeyValue& KeyValue::operator=(std::initializer_list<std::initializer_list<KeyValue>> const& u) {
    m_node_ = make_node(u);
    return *this;
}
KeyValue& KeyValue::operator=(std::initializer_list<std::initializer_list<std::initializer_list<KeyValue>>> const& u) {
    m_node_ = make_node(u);
    return *this;
}

std::istream& DataNode::Parse(std::istream& is) {
    Parse(std::string(std::istreambuf_iterator<char>(is), {}));
    return is;
}
std::ostream& DataNode::Print(std::ostream& os, int indent) const {
    switch (type()) {
        case DN_ENTITY: {
            auto entity = GetEntity();
            if (entity != nullptr) GetEntity()->Print(os, indent + 1);
        } break;
        case DN_ARRAY: {
            bool new_line = this->size() > 1;
            os << "[";
            if (this->size() > 0) {
                this->Get(0)->Print(os, indent + 1);
                for (size_type i = 1, ie = this->size(); i < ie; ++i) {
                    auto v = this->Get(i);
                    os << ", ";
                    //                    if (new_line && v->type() != DataNode::DN_ENTITY) {
                    //                        os << std::endl << std::setw(indent + 1) << " ";
                    //                    }
                    this->Get(i)->Print(os, indent + 1);
                }
            }
            os << "]";
        } break;
        case DN_TABLE: {
            os << "{ ";
            bool is_first = true;
            bool new_line = this->size() > 1;
            this->Foreach([&](auto k, auto v) {
                ASSERT(v != nullptr);
                if (is_first) {
                    is_first = false;
                } else {
                    os << ", ";
                }
                if (new_line) { os << std::endl << std::setw(indent + 1) << " "; }
                FancyPrint(os, k, indent);
                os << " = ";
                v->Print(os, indent + 1);
            });

            if (new_line) { os << std::endl << std::setw(indent) << " "; }
            os << "}";
        } break;
        case DN_FUNCTION:
            os << "<FUNCTION>";
            break;
        default:
            os << "<N/A>";
            break;
    }

    return os;
}
std::ostream& operator<<(std::ostream& os, DataNode const& entry) { return entry.Print(os, 0); }

DataNode::eNodeType DataNode::type() const { return m_type_; }
size_type DataNode::size() const { return m_entity_ == nullptr ? 0 : 1; }
std::shared_ptr<DataNode> DataNode::Duplicate() const { return const_cast<this_type*>(this)->shared_from_this(); }

std::shared_ptr<DataNode> DataNode::CreateNode(eNodeType e_type) const { return DataNode::New(e_type, ""); };
std::shared_ptr<DataNode> DataNode::CreateNode(std::string const& url, eNodeType e_type) {
    auto node = DataNode::New(e_type, "");
    Set(url, node);
    return node;
};

size_type DataNode::Set(std::string const& uri, const std::shared_ptr<DataNode>& v) { return 0; }
size_type DataNode::Add(std::string const& uri, const std::shared_ptr<DataNode>& v) { return 0; }
size_type DataNode::Delete(std::string const& s) { return 0; }
std::shared_ptr<DataNode> DataNode::Get(std::string const& uri) const { return nullptr; }
void DataNode::Foreach(std::function<void(std::string const&, std::shared_ptr<DataNode> const&)> const& f) const {}
size_type DataNode::Set(index_type s, std::shared_ptr<DataNode> const& v) { return Set(std::to_string(s), v); }
size_type DataNode::Add(index_type s, std::shared_ptr<DataNode> const& v) { return Add(std::to_string(s), v); }
size_type DataNode::Delete(index_type s) { return Delete(std::to_string(s)); }
std::shared_ptr<DataNode> DataNode::Get(index_type s) const { return Get(std::to_string(s)); }
size_type DataNode::Add(const std::shared_ptr<DataNode>& v) { return Add(size(), v); }
size_type DataNode::Set(const std::shared_ptr<DataNode>& v) {
    size_type count = 0;
    if (v != nullptr) {
        v->Foreach([&](std::string k, std::shared_ptr<DataNode> const& v) { count += Set(k, v); });
    }
    return count;
}
size_type DataNode::SetValue(KeyValue const& kv) {
    ASSERT(type() == DN_TABLE);
    return Set(kv.m_key_, kv.m_node_);
}
size_type DataNode::SetValue(std::initializer_list<KeyValue> const& u) {
    ASSERT(type() == DN_TABLE);
    size_type count = 0;
    for (auto const& kv : u) { count += Set(kv.m_key_, kv.m_node_); }
    return count;
}
size_type DataNode::SetValue(std::string const& url, KeyValue const& kv) { return Set(url, make_node(kv)); };
size_type DataNode::SetValue(std::string const& url, std::initializer_list<KeyValue> const& kv) {
    return Set(url, make_node(kv));
};
size_type DataNode::SetValue(std::string const& url, std::initializer_list<std::initializer_list<KeyValue>> const& kv) {
    return Set(url, make_node(kv));
};
size_type DataNode::AddValue(std::string const& url, KeyValue const& kv) { return Add(url, make_node(kv)); };
size_type DataNode::AddValue(std::string const& url, std::initializer_list<KeyValue> const& kv) {
    return Add(url, make_node(kv));
};
size_type DataNode::AddValue(std::string const& url, std::initializer_list<std::initializer_list<KeyValue>> const& kv) {
    return Add(url, make_node(kv));
};

//    static std::regex const sub_group_regex(R"(([^/?#]+)/)", std::regex::optimize);
// static std::regex const match_path_regex(R"(^(/?([/\S]+/)*)?([^/]+)?$)", std::regex::optimize);

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
// std::pair<std::string, std::shared_ptr<DataNode>> RecursiveFindNode(std::shared_ptr<DataNode> root, std::string
// uri,
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
//             std::regex_search(pos, end, sub_match_result, sub_group_regex); pos =
//             sub_match_result.suffix().first) {
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

// DataNode& DataNode::operator+=(KeyValue const& u) {
//    Add(u.m_node_);
//    return *this;
//}
// DataNode& DataNode::operator+=(std::initializer_list<KeyValue> const& u) {
//    for (auto const& v : u) { Add(v.m_node_); }
//    return *this;
//}

}  // namespace data {
}  // namespace simpla {
