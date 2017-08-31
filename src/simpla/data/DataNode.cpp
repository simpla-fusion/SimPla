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
DataNode::DataNode(eNodeType etype) : m_type_(etype) {}
DataNode::~DataNode(){};
std::shared_ptr<DataNode> DataNode::New(eNodeType e_type, std::string const& s) {
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
    return res->CreateNode(e_type);
};

KeyValue::KeyValue(std::string k) : m_key_(std::move(k)), m_node_(DataNode::New(DataNode::DN_ENTITY, "")) {
    m_node_->SetEntity(DataLight::New(true));
}
KeyValue::KeyValue(KeyValue const& other) = default;
KeyValue::KeyValue(KeyValue&& other) noexcept  = default;
KeyValue::~KeyValue() = default;
KeyValue& KeyValue::operator=(KeyValue const& other) {
    m_node_ = DataNode::New(DataNode::DN_TABLE, "");
    m_node_->Set(other.m_key_, other.m_node_);
    return *this;
}

template <typename U>
std::shared_ptr<DataNode> make_node(U const& u) {
    auto res = DataNode::New(DataNode::DN_ENTITY, "");
    res->SetEntity(DataLight::New(u));
    return res;
}
std::shared_ptr<DataNode> make_node(std::initializer_list<KeyValue> const& u) {
    auto res = DataNode::New(DataNode::DN_TABLE, "");
    for (auto const& v : u) { res->Set(v.m_key_, v.m_node_); }
    return res;
}
template <typename U>
std::shared_ptr<DataNode> make_node(std::initializer_list<U> const& u) {
    auto res = DataNode::New(DataNode::DN_ENTITY, "");
    //    res->SetEntity(DataLightT<U*>::New(u));
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
            GetEntity()->Print(os, indent + 1);
        } break;
        case DN_ARRAY: {
            bool new_line = this->size() > 1;
            os << "[";
            if (this->size() > 0) {
                this->Get(0)->Print(os, indent + 1);
                for (size_type i = 1, ie = this->size(); i < ie; ++i) {
                    auto v = this->Get(i);
                    os << ", ";
                    if (new_line && v->type() != DataNode::DN_ENTITY) {
                        os << std::endl << std::setw(indent + 1) << " ";
                    }
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
                return 1;
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
size_type DataNode::size() const { return 0; }
std::shared_ptr<DataNode> DataNode::CreateNode(eNodeType e_type) const { return DataNode::New(e_type, ""); };

std::shared_ptr<DataNode> DataNode::CreateEntity(std::shared_ptr<DataEntity> const& v) const {
    auto res = CreateNode(DN_ENTITY);
    res->SetEntity(v);
    return res;
}

size_type DataNode::Set(std::string const& uri, std::shared_ptr<DataNode> const& v) {
    DOMAIN_ERROR;
    return 0;
}
size_type DataNode::Add(std::string const& uri, std::shared_ptr<DataNode> const& v) {
    DOMAIN_ERROR;
    return 0;
}
size_type DataNode::Delete(std::string const& s) {
    DOMAIN_ERROR;
    return 0;
}
std::shared_ptr<DataNode> DataNode::Get(std::string const& uri) const {
    DOMAIN_ERROR;
    return nullptr;
}
size_type DataNode::Foreach(std::function<size_type(std::string, std::shared_ptr<DataNode>)> const& f) const {
    DOMAIN_ERROR;
    return 0;
}
size_type DataNode::Set(size_type s, std::shared_ptr<DataNode> const& v) { return Set(std::to_string(s), v); }
size_type DataNode::Add(size_type s, std::shared_ptr<DataNode> const& v) { return Add(std::to_string(s), v); }
size_type DataNode::Delete(size_type s) { return Delete(std::to_string(s)); }
std::shared_ptr<DataNode> DataNode::Get(size_type s) const { return Get(std::to_string(s)); }
size_type DataNode::Add(std::shared_ptr<DataNode> const& v) { return Add(size(), v); }

std::shared_ptr<DataEntity> DataNode::GetEntity() const { return nullptr; };
size_type DataNode::SetEntity(const std::shared_ptr<DataEntity>&) { return 0; }

size_type DataNode::SetValue(std::string const& url, KeyValue const& v) {
    return Set(url + "/" + v.m_key_, v.m_node_);
};
size_type DataNode::SetValue(std::string const& url, std::initializer_list<KeyValue> const& v) {
    size_type count = 0;
    for (auto const& kv : v) { count += SetValue(url, kv); }
    return count;
};
size_type DataNode::SetValue(std::string const& url, std::initializer_list<std::initializer_list<KeyValue>> const& v) {
    size_type count = 0;
    for (auto const& kv : v) { count += SetValue(url, kv); }
    return count;
};
size_type DataNode::AddValue(std::string const& url, KeyValue const& v) {
    return AddValue(url + "/" + v.m_key_, v.m_node_);
};
size_type DataNode::AddValue(std::string const& url, std::initializer_list<KeyValue> const& v) {
    size_type count = 0;
    for (auto const& kv : v) { count += AddValue(url, kv); }
    return count;
};
size_type DataNode::AddValue(std::string const& url, std::initializer_list<std::initializer_list<KeyValue>> const& v) {
    size_type count = 0;
    for (auto const& kv : v) { count += AddValue(url, kv); }
    return count;
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
