//
// Created by salmon on 17-3-9.
//

#include "DataBackend.h"
#include <iomanip>
#include <string>

#include <regex>
#include "DataBackendMemory.h"
#include "DataEntity.h"
#include "DataTable.h"
#include "backend/DataBackendHDF5.h"
#include "backend/DataBackendLua.h"
#include "simpla/utilities/Factory.h"
namespace simpla {
namespace data {

bool DataBackend::s_RegisterDataBackends_ = DataBackendMemory::_is_registered &&  //
                                            DataBackendHDF5::_is_registered &&    //
                                            DataBackendLua::_is_registered;

/**
*   https://tools.ietf.org/html/rfc3986#page-50
*
*   Appendix B.  Parsing a URI Reference with a Regular Expression
*
*     As the "first-match-wins" algorithm is identical to the "greedy"
*     disambiguation method used by POSIX regular expressions, it is
*     natural and commonplace to use a regular expression for parsing the
*     potential five components of a URI reference.
*
*     The following line is the regular expression for breaking-down a
*     well-formed URI reference into its components.
*
*   ^(([^:/?#]+):)?(//([^/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?
*    12            3  4          5       6  7        8 9
*
*     The numbers in the second line above are only to assist readability;
*     they indicate the reference points for each subexpression (i.e., each
*     paired parenthesis).  We refer to the value matched for subexpression
*     <n> as $<n>.  For example, matching the above expression to
*
*        http://www.ics.uci.edu/pub/ietf/uri/#Related
*
*     results in the following subexpression matches:
*
*        $1 = http:
*        $2 = http
*        $3 = //www.ics.uci.edu
*        $4 = www.ics.uci.edu
*        $5 = /pub/ietf/uri/
*        $6 = <undefined>
*        $7 = <undefined>
*        $8 = #Related
*        $9 = Related
*
*     where <undefined> indicates that the component is not present, as is
*     the case for the query component in the above example.  Therefore, we
*     can determine the value of the five components as
*
*        scheme    = $2
*        authority = $4
*        path      = $5
*        query     = $7
*        fragment  = $9
*
*     Going in the opposite direction, we can recreate a URI reference from
*     its components by using the algorithm of Section 5.3.
*
*   */
static std::regex uri_regex(R"(^(([^:\/?#]+):)?(\/\/([^\/?#]*))?([^?#]*)(\?([^#]*))?(#(.*))?)");
static std::regex file_extension_regex(R"(^(.*)(\.([[:alnum:]]+))$)");
std::shared_ptr<DataBackend> DataBackend::Create(std::string const &uri, std::string const &ext_param) {
    if (uri.empty() || uri == "mem://") { return std::make_shared<DataBackendMemory>(); }

    std::string scheme;
    std::string path = uri;
    std::smatch uri_match_result;
    std::string authority;
    std::string query;
    std::string fragment;
    if (std::regex_match(uri, uri_match_result, uri_regex)) {
        //        for (size_type i = 0, ie = uri_match_result.size(); i < ie; ++i) {
        //            std::cout << i << "\t:" << uri_match_result.str(i) << std::endl;
        //        }
        scheme = (uri_match_result.str(2) == "file") ? "" : uri_match_result.str(2);
        authority = uri_match_result.str(4);
        path = uri_match_result.str(5);
        query = uri_match_result.str(7);
        fragment = uri_match_result.str(9);
    }

    if (scheme.empty()) {
        if (std::regex_match(path, uri_match_result, file_extension_regex)) { scheme = uri_match_result.str(3); }
    }
    if (scheme.empty()) {
        RUNTIME_ERROR << "illegal URI: [" << uri_match_result.str(0) << "|" << uri_match_result.str(1) << "|"
                      << uri_match_result.str(2) << "|" << uri_match_result.str(3) << "]" << std::endl;
    }

    VERBOSE << "Create New Data Backend [ " << scheme << " : " << authority << path << " ]" << std::endl;
    auto res = base_type::Create(scheme);
    ASSERT(res != nullptr);
    res->Connect(authority, path, query, fragment);
    return res;
};

}  // namespace data {
}  // namespace simpla {