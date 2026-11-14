#pragma once
// C++17 shim for CUDA TUs compiled by nvcc 13 (device code is C++17, where
// MSVC removed std::unary_function; core/core.h free_ptr derives from it).
// MSVC C++14 TUs still provide the real template, so this only applies in
// C++17-and-later modes.  Include AFTER the CUDA headers, never via /FI:
// a force-included namespace-std declaration breaks nvcc's prepass.
#if defined(_MSC_VER) && _MSVC_LANG >= 201703L
namespace std
{
template <class Arg, class Result>
struct unary_function
{
	typedef Arg argument_type;
	typedef Result result_type;
};
}  // namespace std
#endif
