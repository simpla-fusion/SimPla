//
// Created by salmon on 17-3-8.
//

#ifndef SIMPLA_DATAENTITYFACTROY_H
#define SIMPLA_DATAENTITYFACTROY_H

#include <simpla/utilities/nTuple.h>
#include <memory>
#include <string>

namespace simpla {
namespace data {
class DataEntity;
/**
 *  PUT and POST are both unsafe methods. However, PUT is idempotent, while POST is not.
 *
 *  HTTP/1.1 SPEC
 *  @quota
 *   The POST method is used to request that the origin server accept the entity enclosed in
 *   the request as a new subordinate of the resource identified by the Request-URI in the Request-Line
 *
 *  @quota
 *  The PUT method requests that the enclosed entity be stored under the supplied Request-URI.
 *  If the Request-URI refers to an already existing resource, the enclosed entity SHOULD be considered as a
 *  modified version of the one residing on the origin server. If the Request-URI does not point to an existing
 *  resource, and that URI is capable of being defined as a new resource by the requesting user agent, the origin
 *  server can create the resource with that URI."
 *
 */

namespace traits {
template <typename U>
struct is_light_data
    : public std::integral_constant<bool, std::is_arithmetic<U>::value || std::is_same<U, bool>::value> {};

template <>
struct is_light_data<std::string> : public std::integral_constant<bool, true> {};
template <>
struct is_light_data<char const*> : public std::integral_constant<bool, true> {};

}  // namespace traits {

}  // namespace data {
}  // namespace simpla {
#endif  // SIMPLA_DATAENTITYFACTROY_H
