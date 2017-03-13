//
// Created by salmon on 17-3-9.
//

#include "DataBackend.h"
#include <iomanip>
#include <string>

#include <simpla/toolbox/ParserURI.h>
#include <regex>
#include "DataBackendMemory.h"
#include "DataEntity.h"
#include "DataTable.h"
namespace simpla {
namespace data {

DataBackendFactory::DataBackendFactory() : base_type() { RegisterDefault(); };
DataBackendFactory::~DataBackendFactory(){};

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
static std::regex file_extension_regex(R"(^(.*)\.([[:alnum:]]+)$)");
std::shared_ptr<DataBackend> DataBackendFactory::Create(std::string const &uri, std::string const &ext_param) {
    if (uri == "" || uri == "mem://") { return std::make_shared<DataBackendMemory>(); }

    std::string scheme = "";
    std::string path = uri;
    std::smatch uri_match_result;

    if (std::regex_match(uri, uri_match_result, uri_regex)) {
        //        for (size_type i = 0, ie = uri_match_result.size(); i < ie; ++i) {
        //            std::cout << i << "\t:" << uri_match_result.str(i) << std::endl;
        //        }
        scheme = (uri_match_result.str(2) == "file") ? "" : uri_match_result.str(2);
        path = uri_match_result.str(5);
    }

    if (scheme == "") {
        if (std::regex_match(path, uri_match_result, file_extension_regex)) { scheme = uri_match_result.str(2); }
    }
    if (scheme == "") { RUNTIME_ERROR << "illegal URI: [" << uri << "]" << std::endl; }

    LOGGER << "CreateNew  DataBackend [ " << scheme << " : " << path << "]" << std::endl;
    std::shared_ptr<DataBackend> res{base_type::Create(scheme)};
    res->Connect(path);
    return res;
};

std::vector<std::string> DataBackendFactory::GetBackendList() const {
    std::vector<std::string> res;
    for (auto const &item : *this) { res.push_back(item.first); }
    return std::move(res);
};

}  // namespace data {
}  // namespace simpla {