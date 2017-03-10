//
// Created by salmon on 17-3-6.
//
#include "DataBackendFactory.h"
#include <iomanip>
#include <regex>
#include "DataBackend.h"
#include "DataBackendMemory.h"
namespace simpla {
namespace data {
struct DataBackendFactory::pimpl_s {
    static std::regex url_regex;
    std::tuple<std::string, std::string, std::string, std::string, std::string> Parse(std::string const &url) const;
};
std::regex DataBackendFactory::pimpl_s::url_regex(
    R"(^(([^:\/?#]+):)?(//([^\/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?)", std::regex::extended);

DataBackendFactory::DataBackendFactory() : base_type(), m_pimpl_(new pimpl_s) { RegisterDefault(); };
DataBackendFactory::~DataBackendFactory(){};

std::ostream &DataBackendFactory::Print(std::ostream &os, int indent) const {
    os << std::setw(indent) << " "
       << "{";
    for (auto const &item : *this) { os << item.first << ", "; }
    os << "}";
    return os;
};

/**
 *  @quota https://tools.ietf.org/html/rfc3986#page-50
 *
 *     ^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?
 *      12            3  4          5       6  7        8 9
 *
 *  The numbers in the second line above are only to assist readability;
 *  they indicate the reference points for each subexpression (i.e., each
 *  paired parenthesis).  We refer to the value matched for subexpression
 *  <n> as $<n>.  For example, matching the above expression to
 *
 *     http://www.ics.uci.edu/pub/ietf/uri/#Related
 *
 *  results in the following subexpression matches:
 *
 *     $1 = http:
 *     $2 = http
 *     $3 = //www.ics.uci.edu
 *     $4 = www.ics.uci.edu
 *     $5 = /pub/ietf/uri/
 *     $6 = <undefined>
 *     $7 = <undefined>
 *     $8 = #Related
 *     $9 = Related
 *
 *  where <undefined> indicates that the component is not present, as is
 *  the case for the query component in the above example.  Therefore, we
 *  can determine the value of the five components as
 *
 *     scheme    = $2
 *     authority = $4
 *     path      = $5
 *     query     = $7
 *     fragment  = $9
 *
 *  Going in the opposite direction, we can recreate a URI reference from
 *  its components by using the algorithm of Section 5.3.
 */

std::tuple<std::string, std::string, std::string, std::string, std::string> DataBackendFactory::pimpl_s::Parse(
    std::string const &url) const {
    unsigned counter = 0;

    std::smatch url_match_result;

    std::cout << "Checking: " << url << std::endl;

    if (std::regex_match(url, url_match_result, url_regex)) {
        return std::make_tuple(url_match_result[2], url_match_result[4], url_match_result[5], url_match_result[7],
                               url_match_result[8]);
    } else {
        RUNTIME_ERROR << "URI mismatch!" << std::endl;
    }
}

DataBackend *DataBackendFactory::Create(std::string const &uri) {
    DataBackend *res = nullptr;
    if (uri == "") {
        res = new DataBackendMemory();
    } else {
        std::smatch url_match_result;

        std::cout << "Checking: " << uri << std::endl;

        if (std::regex_match(uri, url_match_result, m_pimpl_->url_regex)) {
            std::string scheme;
            std::string authority;
            std::string path;
            std::string query;
            std::string fragment;

            res = base_type::Create(scheme);
            if (res == nullptr) {
                res = new DataBackendMemory();
            } else {
                res->Open(authority, path, query);
            }
        }
        //        size_type pos = uri.find_last_of('.');
        //        if (pos != std::string::npos) { ext = uri.substr(pos + 1); }
    }
    return res;
}

}  // namespace data{
}  // namespace simpla{