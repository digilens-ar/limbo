
#include <limbo/kernel.hpp>
#include <limbo/model.hpp>
#include <limbo/tools/macros.hpp>
#include <limbo/serialize/text_archive.hpp>
#include "FunctionExport.hpp"

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
        };

        struct opt_irpropplus : defaults::opt_irpropplus
        {
            BO_PARAM(double, min_gradient, 1e-3);
        };
    };
}

//Export loglikelihood
int main()
{
    using Kernel = kernel::SquaredExpARD<Params::kernel, Params::kernel_squared_exp_ard>;
    using Model = model::GaussianProcess<Kernel, mean::Data, model::gp::KernelLFOpt<opt::Irpropplus<Params::opt_irpropplus>>>;

    std::filesystem::path rootDir(R"(C:\Users\NicholasAnthony\source\repos\wt_gui\external\WaveTracer\testing\optimizerTests\resources\DIA_highD\optimizations\hp_10_irpropplus_ser_1eneg3)");
    Model m = Model::load(serialize::TextArchive((rootDir / "modelArchive_init").string()));

    serialize::exportFunction(
        rootDir / "pre",
        serialize::MeanFunction | serialize::KernelFunction | serialize::LogLikelihood,
        m,
        6);

    m.optimize_hyperparams(); // TODO log each iteration
    // serialize::FunctionExport(
    //     rootDir / "post",
    //     serialize::FunctionExport::MeanFunction | serialize::FunctionExport::KernelFunction | serialize::FunctionExport::LogLikelihood,
    //     m,
    //     200);
    return 0;
}