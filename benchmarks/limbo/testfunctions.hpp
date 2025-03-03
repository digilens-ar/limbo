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
#define _USE_MATH_DEFINES
#include <Eigen/Core>

// support functions
inline double sign(double x)
{
    if (x < 0)
        return -1;
    if (x > 0)
        return 1;
    return 0;
}

inline double sqr(double x)
{
    return x * x;
};

inline double hat(double x)
{
    if (x != 0)
        return std::log(std::abs(x));
    return 0;
}

inline double c1(double x)
{
    if (x > 0)
        return 10;
    return 5.5;
}

inline double c2(double x)
{
    if (x > 0)
        return 7.9;
    return 3.1;
}

inline Eigen::VectorXd t_osz(const Eigen::VectorXd& x)
{
    Eigen::VectorXd r = x;
    for (size_t i = 0; i < static_cast<size_t>(x.size()); i++)
        r(i) = sign(x(i)) * std::exp(hat(x(i)) + 0.049 * std::sin(c1(x(i)) * hat(x(i))) + std::sin(c2(x(i)) * hat(x(i))));
    return r;
}

struct Sphere {
    BO_PARAM(size_t, dim_in, 2);


    double operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd opt(2);
        opt << 0.5, 0.5;
        return (x - opt).squaredNorm();
    }

    Eigen::MatrixXd solutions() const
    {
        Eigen::MatrixXd sols(1, 2);
        sols << 0.5, 0.5;
        return sols;
    }
};

struct Ellipsoid {
    BO_PARAM(size_t, dim_in, 2);


    double operator()(const Eigen::VectorXd& x) const
    {
        Eigen::VectorXd opt(2);
        opt << 0.5, 0.5;
        Eigen::VectorXd z = t_osz(x - opt);
        double r = 0;
        for (size_t i = 0; i < dim_in(); ++i)
            r += std::pow(10., ((double)i) / (dim_in() - 1.0)) * z(i) * z(i) + 1;
        return r;
    }

    Eigen::MatrixXd solutions() const
    {
        Eigen::MatrixXd sols(1, 2);
        sols << 0.5, 0.5;
        return sols;
    }
};

struct Rastrigin {
    BO_PARAM(size_t, dim_in, 4);


    double operator()(const Eigen::VectorXd& xx) const
    {
        Eigen::VectorXd x = xx;
        for (size_t i = 0; i < static_cast<size_t>(x.size()); i++)
            x(i) = 2. * xx(i) - 1.;
        double f = 10. * dim_in();
        for (size_t i = 0; i < dim_in(); ++i)
            f += x(i) * x(i) - 10. * std::cos(2 * M_PI * x(i));
        return f;
    }

    Eigen::MatrixXd solutions() const
    {
        Eigen::MatrixXd sols = Eigen::MatrixXd::Zero(1, 4);
        for (size_t i = 0; i < 4; i++)
            sols(0, i) = (sols(0, i) + 1.) / 2.;
        return sols;
    }
};

// see : http://www.sfu.ca/~ssurjano/hart3.html
struct Hartmann3 {
    BO_PARAM(size_t, dim_in, 3);


    double operator()(const Eigen::VectorXd& x) const
    {
        Eigen::MatrixXd a(4, 3);
        Eigen::MatrixXd p(4, 3);
        a << 3.0, 10., 30.,
            0.1, 10., 35.,
            3.0, 10., 30.,
            0.1, 10., 35.;
        p << 0.3689, 0.1170, 0.2673,
            0.4699, 0.4387, 0.7470,
            0.1091, 0.8732, 0.5547,
            0.0381, 0.5743, 0.8828;
        Eigen::VectorXd alpha(4);
        alpha << 1.0, 1.2, 3.0, 3.2;

        double res = 0.;
        for (size_t i = 0; i < 4; i++) {
            double s = 0.;
            for (size_t j = 0; j < 3; j++) {
                s += a(i, j) * sqr(x(j) - p(i, j));
            }
            res += alpha(i) * std::exp(-s);
        }
        return -res;
    }

    Eigen::MatrixXd solutions() const
    {
        Eigen::MatrixXd sols(1, 3);
        sols << 0.114614, 0.555649, 0.852547;
        return sols;
    }
};

// see : http://www.sfu.ca/~ssurjano/hart6.html
struct Hartmann6 {
    BO_PARAM(size_t, dim_in, 6);


    double operator()(const Eigen::VectorXd& x) const
    {
        Eigen::MatrixXd a(4, 6);
        Eigen::MatrixXd p(4, 6);
        a << 10., 3., 17., 3.5, 1.7, 8.,
            0.05, 10., 17., 0.1, 8., 14.,
            3., 3.5, 1.7, 10., 17., 8.,
            17., 8., 0.05, 10., 0.1, 14.;
        p << 0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886,
            0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991,
            0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650,
            0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381;

        Eigen::VectorXd alpha(4);
        alpha << 1.0, 1.2, 3.0, 3.2;

        double res = 0.;
        for (size_t i = 0; i < 4; i++) {
            double s = 0.;
            for (size_t j = 0; j < 6; j++) {
                s += a(i, j) * sqr(x(j) - p(i, j));
            }
            res += alpha(i) * std::exp(-s);
        }
        return -res;
    }

    Eigen::MatrixXd solutions() const
    {
        Eigen::MatrixXd sols(1, 6);
        sols << 0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573;
        return sols;
    }
};

// see : http://www.sfu.ca/~ssurjano/goldpr.html
// (with ln, as suggested in Jones et al.)
struct GoldsteinPrice {
    BO_PARAM(size_t, dim_in, 2);


    double operator()(const Eigen::VectorXd& xx) const
    {
        Eigen::VectorXd x = xx;
        for (size_t i = 0; i < static_cast<size_t>(x.size()); i++)
            x(i) = 4. * xx(i) - 2.;

        double fact1a = sqr(x(0) + x(1) + 1.);
        double fact1b = 19. - 14. * x(0) + 3. * sqr(x(0)) - 14. * x(1) + 6. * x(0) * x(1) + 3. * sqr(x(1));
        double fact1 = 1. + fact1a * fact1b;

        double fact2a = sqr(2. * x(0) - 3. * x(1));
        double fact2b = 18. - 32. * x(0) + 12. * sqr(x(0)) + 48. * x(1) - 36. * x(0) * x(1) + 27. * sqr(x(1));
        double fact2 = 30. + fact2a * fact2b;

        double r = fact1 * fact2;

        return (std::log(r) - 8.693) / 2.427;
        // return std::log(r) - 5.;
    }

    Eigen::MatrixXd solutions() const
    {
        Eigen::MatrixXd sols(1, 2);
        sols << 0.5, 0.25;
        return sols;
    }
};

struct BraninNormalized {
    BO_PARAM(size_t, dim_in, 2);


    double operator()(const Eigen::VectorXd& x) const
    {
        double x1 = x(0) * 15 - 5;
        double x2 = x(1) * 15;

        double term1 = sqr(x2 - (5.1 * sqr(x1) / (4. * sqr(M_PI))) + 5. * x1 / M_PI - 6);
        double term2 = (10. - 10. / (8. * M_PI)) * std::cos(x1);

        return (term1 + term2 - 44.81) / 51.95;
    }

    Eigen::MatrixXd solutions() const
    {
        Eigen::MatrixXd sols(3, 2);
        sols << - M_PI, 12.275,
            M_PI, 2.275,
            9.42478, 2.475;

        sols(0, 0) = (sols(0, 0) + 5.) / 15.;
        sols(1, 0) = (sols(1, 0) + 5.) / 15.;
        sols(2, 0) = (sols(2, 0) + 5.) / 15.;
        sols(0, 1) = sols(0, 1) / 15.;
        sols(1, 1) = sols(1, 1) / 15.;
        sols(2, 1) = sols(2, 1) / 15.;
        return sols;
    }
};

struct SixHumpCamel {
    BO_PARAM(size_t, dim_in, 2);


    double operator()(const Eigen::VectorXd& x) const
    {
        double x1 = -3 + 6 * x(0);
        double x2 = -2 + 4 * x(1);
        double x1_2 = sqr(x1);
        double x2_2 = sqr(x2);

        double tmp1 = (4 - 2.1 * x1_2 + sqr(x1_2) / 3.) * x1_2;
        double tmp2 = x1 * x2;
        double tmp3 = (-4 + 4 * x2_2) * x2_2;
        return tmp1 + tmp2 + tmp3;
    }

    Eigen::MatrixXd solutions() const
    {
        Eigen::MatrixXd sols(2, 2);
        sols << 0.0898, -0.7126,
            -0.0898, 0.7126;
        sols(0, 0) = (sols(0, 0) + 3.) / 6.;
        sols(1, 0) = (sols(1, 0) + 3.) / 6.;
        sols(0, 1) = (sols(0, 1) + 2.) / 4.;
        sols(1, 1) = (sols(1, 1) + 2.) / 4.;
        return sols;
    }
};

template<typename T>
concept TestFunction = requires (T a)
{
    { a.dim_in() } -> std::convertible_to<int>;
    { a(Eigen::VectorXd{}) } -> std::convertible_to<double>;
    { a.solutions() } -> std::convertible_to<Eigen::MatrixXd>;
};

template <TestFunction Function>
class Benchmark {
public:
    BO_PARAM(size_t, dim_in, Function::dim_in());

    std::tuple<limbo::EvaluationStatus, double> operator()(const Eigen::VectorXd& x) const
    {
        return {limbo::EvaluationStatus::OK, -f(x)}; // return the negative since the test functions are originally intended for minimization but we do maximization
    }

    double accuracy(double x)
    {
        Eigen::MatrixXd sols = f.solutions();
        double diff = std::abs(x + f(sols.row(0)));
        double min_diff = diff;

        for (int i = 1; i < sols.rows(); i++) {
            diff = std::abs(x + f(sols.row(i)));
            if (diff < min_diff)
                min_diff = diff;
        }

        return min_diff;
    }

    Function f;
};


static_assert(limbo::concepts::StateFunc<Benchmark<Hartmann3>>);