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
#ifndef LIMBO_TOOLS_PARALLEL_HPP
#define LIMBO_TOOLS_PARALLEL_HPP

#include <algorithm>
#include <vector>

#ifdef LIMBO_USE_TBB
// Quick hack for definition of 'I' in <complex.h>
#undef I
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_reduce.h>
#endif

///@defgroup par_tools

namespace limbo::tools::par {
    ///@ingroup par_tools
    /// parallel for
    template <typename F>
    void loop(size_t begin, size_t end, const F& f)
    {
#ifdef LIMBO_USE_TBB
        tbb::parallel_for(size_t(begin), end, size_t(1), [&](size_t i) {
            f(i);
        });
#else
        for (size_t i = begin; i < end; ++i)
            f(i);
#endif
    }

    /// @ingroup par_tools
    /// parallel for_each
    template <typename Iterator, typename F>
    void for_each(Iterator begin, Iterator end, const F& f)
    {
#ifdef LIMBO_USE_TBB
        tbb::parallel_for_each(begin, end, f);
#else
        for (Iterator i = begin; i != end; ++i)
            f(*i);
#endif
    }

    /// @ingroup par_tools
    /// parallel max
    template <typename T, typename F, typename C>
    T max(const T& init, int num_steps, const F& f, const C& comp)
    {
#ifdef LIMBO_USE_TBB
        auto body = [&](const tbb::blocked_range<size_t>& r, T current_max) -> T {
            
            for (size_t i = r.begin(); i != r.end(); ++i)
            {
                T v = f(i);
                if (comp(v, current_max))
                  current_max = v;
            }
            return current_max;
                
        };
        auto joint = [&](const T& p1, const T& p2) -> T {
            if (comp(p1, p2))
                return p1;
            return p2;
        };
        return tbb::parallel_reduce(tbb::blocked_range<size_t>(0, num_steps), init, body, joint);
#else
        T current_max = init;
        for (int i = 0; i < num_steps; ++i) {
            T v = f(i);
            if (comp(v, current_max))
                current_max = v;
        }
        return current_max;
#endif
    }

}

#endif
