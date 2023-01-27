//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Konstantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
//|   - Roberto Rama (bertoski@gmail.com)
//|
//| This software is a computer library whose purpose is to optimize continuous,
//| black-box functions. It mainly implements Gaussian processes and Bayesian
//| optimization.
//| Main repository: http://github.com/resibots/limbo
//| Documentation: http://www.resibots.eu/limbo
//|
//| This software is governed by the CeCILL-C license under French law and
//| abiding by the rules of distribution of free software.  You can  use,
//| modify and/ or redistribute the software under the terms of the CeCILL-C
//| license as circulated by CEA, CNRS and INRIA at the following URL
//| "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and  rights to copy,
//| modify and redistribute granted by the license, users are provided only
//| with a limited warranty  and the software's author,  the holder of the
//| economic rights,  and the successive licensors  have only  limited
//| liability.
//|
//| In this respect, the user's attention is drawn to the risks associated
//| with loading,  using,  modifying and/or developing or reproducing the
//| software by the user in light of its specific status of free software,
//| that may mean  that it is complicated to manipulate,  and  that  also
//| therefore means  that it is reserved for developers  and  experienced
//| professionals having in-depth computer knowledge. Users are therefore
//| encouraged to load and test the software's suitability as regards their
//| requirements in conditions enabling the security of their systems and/or
//| data to be ensured and,  more generally, to use and operate it in the
//| same conditions as regards security.
//|
//| The fact that you are presently reading this means that you have had
//| knowledge of the CeCILL-C license and that you accept its terms.
//|
#ifndef LIMBO_TOOLS_MACROS_HPP
#define LIMBO_TOOLS_MACROS_HPP

#include <Eigen/Core>
#include <iostream>
#include <regex>

#define BO_PARAM(Type, Name, Value) \
    static constexpr auto Name = []()constexpr -> Type { return Value; };

#define BO_REQUIRED_PARAM(Type, Name)                                         \
    static const Type Name()                                                  \
    {                                                                         \
        static_assert(false, "You need to define the parameter:" #Name " !"); \
        return Type();                                                        \
    }

#define BO_DYN_PARAM(Type, Name)           \
    static Type _##Name;                   \
    static Type Name() { return _##Name; } \
    static void set_##Name(const Type& v) { _##Name = v; }

#define BO_DECLARE_DYN_PARAM(Type, Namespace, Name) Type Namespace::_##Name;

#define NUMARGS(...) std::tuple_size<decltype(std::make_tuple(__VA_ARGS__))>::value

#define BO_PARAM_ARRAY(Type, Name, ...)                  \
    static Type Name(size_t i)                           \
    {                                                    \
        assert(i < NUMARGS(__VA_ARGS__));            \
        static constexpr Type _##Name[] = {__VA_ARGS__}; \
        return _##Name[i];                               \
    }                                                    \
    static constexpr size_t Name##_size()                \
    {                                                    \
        return NUMARGS(__VA_ARGS__);                 \
    }                                                    \
    using Name##_t = Type;

#define BO_PARAM_VECTOR(Type, Name, ...)                                                    \
    static const Eigen::Matrix<Type, NUMARGS(__VA_ARGS__), 1> Name()                    \
    {                                                                                       \
        static constexpr Type _##Name[] = {__VA_ARGS__};                                    \
        return Eigen::Map<const Eigen::Matrix<Type, NUMARGS(__VA_ARGS__), 1>>(_##Name); \
    }

#define BO_PARAM_STRING(Name, Value) \
    static constexpr const char* Name() { return Value; }

#define BO_PARAMS(Stream, P)                                  \
    struct Ps__ {                                             \
        Ps__()                                                \
        {                                                     \
            static std::string __params = #P;                 \
			std::regex_replace(__params, std::regex(";"), ";\n"); \
			std::regex_replace(__params, std::regex("{"), "{\n"); \
            Stream << "Parameters:" << __params << std::endl; \
        }                                                     \
    };                                                        \
    P;                                                        \
    static Ps__ ____p;

#endif
