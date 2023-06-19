
#include <limbo/kernel.hpp>
#include <limbo/model.hpp>
#include <limbo/tools/macros.hpp>
#include <limbo/serialize/text_archive.hpp>
#include "function_export.hpp"

using namespace limbo;

namespace {
    struct Params {
        struct kernel : defaults::kernel {
            BO_PARAM(double, noise, 0.0);
        };

        struct kernel_squared_exp_ard : defaults::kernel_squared_exp_ard {};

        struct kernel_exp : defaults::kernel_exp {};

        struct kernel_maternthreehalves : defaults::kernel_maternthreehalves {};

        struct kernel_maternfivehalves : defaults::kernel_maternfivehalves {};

        struct opt_rprop : defaults::opt_rprop
        {
            BO_PARAM(double, eps_stop, 0);
            // BO_PARAM(double, eps_stop, 1e-10);
        };
    };
}

//Export loglikelihood
int main()
{
    using Kernel = kernel::SquaredExpARD<Params::kernel, Params::kernel_squared_exp_ard>;
    using Model = model::GP<Kernel, mean::Data, model::gp::KernelLFOpt<Params::opt_rprop>>;

    std::filesystem::path rootDir(LIMBO_TEST_RESOURCES_DIR);
    Model m = Model::load(serialize::TextArchive((rootDir / "modelArchive_3d_init").string()));

    serialize::FunctionExport(
        rootDir / "pre",
        serialize::FunctionExport::MeanFunction | serialize::FunctionExport::KernelFunction | serialize::FunctionExport::LogLikelihood,
        m,
        100);

    m.optimize_hyperparams(); // TODO log each iteration
    // serialize::FunctionExport(
    //     rootDir / "post",
    //     serialize::FunctionExport::MeanFunction | serialize::FunctionExport::KernelFunction | serialize::FunctionExport::LogLikelihood,
    //     m,
    //     200);
    return 0;
}