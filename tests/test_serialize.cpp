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

#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <limbo/kernel/exp.hpp>
#include <limbo/mean/constant.hpp>
#include <limbo/mean/function_ard.hpp>
#include <limbo/mean/null_function.hpp>
#include <limbo/tools/random.hpp>
#include <limbo/model/gp.hpp>
#include <limbo/model/gp/mean_lf_opt.hpp>
#include <limbo/model/multi_gp.hpp>
#include <limbo/serialize/binary_archive.hpp>
#include <limbo/serialize/text_archive.hpp>

#include "limbo/opt/rprop.hpp"

#ifndef LIMBO_TEST_TEMP_DIR
#error "Please define LIMBO_TEST_TEMP_DIR to a folder to be used for temporary storage"
#endif

namespace {
    struct Params {
        struct kernel_exp {
            BO_PARAM(double, sigma_sq, 1.0);
            BO_PARAM(double, l, 0.2);
        };
        struct kernel : public limbo::defaults::kernel {
        };
        struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
        };
        struct opt_irpropplus : public limbo::defaults::opt_irpropplus {
        };

        struct kernel_maternfivehalves {
            BO_PARAM(double, sigma_sq, 1);
            BO_PARAM(double, l, 1);
        };

        struct mean_constant {
            BO_PARAM(double, constant, 1);
        };
    };
}

// Different parameters in load to test
struct LoadParams {
    struct kernel_exp {
        BO_PARAM(double, sigma_sq, 10.0);
        BO_PARAM(double, l, 1.);
    };
    struct kernel : public limbo::defaults::kernel {
    };
    struct kernel_squared_exp_ard : public limbo::defaults::kernel_squared_exp_ard {
        BO_PARAM(double, sigma_sq, 10.0);
    };
    struct opt_irpropplus : public limbo::defaults::opt_irpropplus {
    };

    struct kernel_maternfivehalves {
        BO_PARAM(double, sigma_sq, 2.);
        BO_PARAM(double, l, 0.1);
    };

    struct mean_constant {
        BO_PARAM(double, constant, -1);
    };
};

template <typename GP, typename GPLoad, typename Archive>
void test_gp(const std::string& name, bool optimize_hp = true)
{
    using namespace limbo;

    // our data (3-D inputs, 1-D outputs)
    std::vector<Eigen::VectorXd> samples;
    std::vector<double> observations;

    size_t n = 8;
    for (size_t i = 0; i < n; i++) {
        Eigen::VectorXd s = tools::random_vector(3).array() * 4.0 - 2.0;
        samples.push_back(s);
        observations.push_back(std::cos(s(0) * s(1) * s(2)));
    }
    // 3-D inputs, 1-D outputs
    GP gp = GP::createFromSamples(std::move(samples), std::move(observations));
    if (optimize_hp)
        gp.optimize_hyperparams();

    // attempt to save
    Archive a1(name);
    gp.save(a1);
    // We can also save like this
    // gp.template save<Archive>(name);

    // attempt to load -- use only the name
    GPLoad gp2 = GPLoad::load(Archive(name));

    GTEST_ASSERT_EQ(gp.samples().size(), gp2.samples().size());

    // check that the two GPs make the same predictions
    size_t k = 1000;
    for (size_t i = 0; i < k; i++) {
        Eigen::VectorXd s = tools::random_vector(3).array() * 4.0 - 2.0;
        auto [mu1, sigma_sq_1] = gp.query(s);
        auto [mu2, sigma_sq_2] = gp2.query(s);
        if constexpr (requires (GP const& gp) { gp.kernel_function(); }) {
            ASSERT_EQ(gp.kernel_function().noise(), gp2.kernel_function().noise());
        }
        GTEST_ASSERT_LE(mu1 - mu2, 1e-10);
        if constexpr (std::is_same_v<decltype(sigma_sq_1), double>) {
            ASSERT_NEAR(sigma_sq_1, sigma_sq_2, 1e-10);
        }
        else
        { // is an eigen vector
            GTEST_ASSERT_LE((sigma_sq_1 - sigma_sq_2).norm(), 1e-10);
        }
    }

    // attempt to load without recomputing
    // and without knowing the dimensions
    Archive a3(name);
    auto gp3 = GPLoad::load(a3, false);

    GTEST_ASSERT_EQ(gp.samples().size(), gp3.samples().size());

    // check that the two GPs make the same predictions
    for (size_t i = 0; i < k; i++) {
        Eigen::VectorXd s = tools::random_vector(3).array() * 4.0 - 2.0;
        auto [mu1, sigma_sq_1] = gp.query(s);
        auto [mu2, sigma_sq_2] = gp3.query(s);
        if constexpr (requires (GP const& gp) { gp.kernel_function(); }) {
            ASSERT_EQ(gp.kernel_function().noise(), gp3.kernel_function().noise());
        }
        GTEST_ASSERT_LE(mu1 - mu2, 1e-10);
        if constexpr (std::is_same_v<decltype(sigma_sq_1), double>) {
            ASSERT_NEAR(sigma_sq_1, sigma_sq_2, 1e-10);
        }
        else
        { // is an eigen vector
            GTEST_ASSERT_LE((sigma_sq_1 - sigma_sq_2).norm(), 1e-10);
        }
    }
}

static std::string rootDir(LIMBO_TEST_TEMP_DIR);

TEST(Limbo_Serialize, text_archive)
{
    std::cout << Params::kernel::noise() << "\n";
    test_gp<limbo::model::GPOpt<Params>, limbo::model::GPOpt<LoadParams>, limbo::serialize::TextArchive>(rootDir + "/gp_opt_text");
    test_gp<limbo::model::GPBasic<Params>, limbo::model::GPBasic<LoadParams>, limbo::serialize::TextArchive>(rootDir + "/gp_basic_text", false);

    using GPMean = limbo::model::GaussianProcess<limbo::kernel::MaternFiveHalves<Params::kernel, Params::kernel_maternfivehalves>, limbo::mean::Constant<Params::mean_constant>, limbo::model::gp::MeanLFOpt<limbo::opt::Irpropplus<Params::opt_irpropplus>>>;
    using GPMeanLoad = limbo::model::GaussianProcess<limbo::kernel::MaternFiveHalves<LoadParams::kernel, LoadParams::kernel_maternfivehalves>, limbo::mean::Constant<LoadParams::mean_constant>, limbo::model::gp::MeanLFOpt<limbo::opt::Irpropplus<LoadParams::opt_irpropplus>>>;
    test_gp<GPMean, GPMeanLoad, limbo::serialize::TextArchive>(rootDir + "/gp_mean_text");
}

TEST(Limbo_Serialize, bin_archive)
{
    test_gp<limbo::model::GPOpt<Params>, limbo::model::GPOpt<LoadParams>, limbo::serialize::BinaryArchive>(rootDir + "/gp_opt_bin");
    test_gp<limbo::model::GPBasic<Params>, limbo::model::GPBasic<LoadParams>, limbo::serialize::BinaryArchive>(rootDir + "/gp_basic_bin", false);

    using GPMean = limbo::model::GaussianProcess<limbo::kernel::MaternFiveHalves<Params::kernel, Params::kernel_maternfivehalves>, limbo::mean::Constant<Params::mean_constant>, limbo::model::gp::MeanLFOpt<limbo::opt::Irpropplus<Params::opt_irpropplus>>>;
    using GPMeanLoad = limbo::model::GaussianProcess<limbo::kernel::MaternFiveHalves<LoadParams::kernel, LoadParams::kernel_maternfivehalves>, limbo::mean::Constant<LoadParams::mean_constant>, limbo::model::gp::MeanLFOpt<limbo::opt::Irpropplus<LoadParams::opt_irpropplus>>>;
    test_gp<GPMean, GPMeanLoad, limbo::serialize::BinaryArchive>(rootDir + "/gp_mean_bin");
}

