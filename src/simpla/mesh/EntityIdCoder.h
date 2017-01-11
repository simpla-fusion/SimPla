//
// Created by salmon on 17-1-11.
//

#ifndef SIMPLA_ENTITYIDCODER_H
#define SIMPLA_ENTITYIDCODER_H

namespace simpla {
namespace mesh {

template <int... I>
using v_seq = integer_sequence<int, I...>;
/**
 *               6 ------------------7
 *              /|                  /|
 *             / |                 / |
 *            /  |                /  |
 *          4 ---+-------------- 5   |
 *          |    |               |   |
 *          |    2---------------+---3
 *          |   /                |  /
 *          |  /                 | /
 *          | /                  |/
 *          0 ------------------ 1
 *
 */

template <int... N>
struct PlaceHolder;
template <int N>
struct PlaceHolder<N> {
    int v = 0;

    PlaceHolder<N> operator-(int n) const { return PlaceHolder<N>{v - n}; }
};
static constexpr PlaceHolder<0> I{0};
static constexpr PlaceHolder<1> J{0};
static constexpr PlaceHolder<2> K{0};
PlaceHolder<0> operator""_p(unsigned long long n) { return PlaceHolder<0>{static_cast<int>(n)}; }
static constexpr EX = v_seq<0, 0b001>;  // 1 >> 1 = 0b00
static constexpr EY = v_seq<0, 0b010>;  // 2 >> 1 = 0b01
static constexpr EZ = v_seq<0, 0b100>;  // 4 >> 1 = 0b10

static constexpr FX = v_seq<0, 0b010, 0b110, 0b100>;  // 0+2+6+4 >>2 = 0b110
static constexpr FY = v_seq<0, 0b001, 0b101, 0b100>;
static constexpr FZ = v_seq<0, 0b001, 0b011, 0b010>;
}  // namespace mesh{
}  // namespace simpla{

#endif  // SIMPLA_ENTITYIDCODER_H
