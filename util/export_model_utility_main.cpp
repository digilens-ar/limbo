#include <tbb/tbbmalloc_proxy.h> // Huge improvement in performance with this allocator.
#include <limbo/kernel.hpp>
#include <limbo/model.hpp>
#include <limbo/tools/macros.hpp>
#include <limbo/serialize/text_archive.hpp>
#include <limbo/serialize/binary_archive.hpp>
#include <spdlog/sinks/wincolor_sink.h>
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

void configureLogging()
{
    auto console_sink = std::make_shared<spdlog::sinks::wincolor_stdout_sink_mt>();
    console_sink->set_level(spdlog::level::info);
    console_sink->set_pattern("[%^%l%$] %v");

    auto logger = std::make_shared<spdlog::logger>("default", spdlog::sinks_init_list{ console_sink });
    logger->set_level(spdlog::level::info);
    spdlog::set_default_logger(logger);
    spdlog::flush_on(spdlog::level::info);
}

//Export loglikelihood
int main()
{
    configureLogging();

    using Kernel = kernel::SquaredExpARD<Params::kernel, Params::kernel_squared_exp_ard>;
    using Model = model::GaussianProcess<Kernel, mean::Data, model::gp::KernelLFOpt<opt::Irpropplus<Params::opt_irpropplus>>>;

    std::filesystem::path rootDir(R"(C:\Users\NicholasAnthony\source\repos\wt_gui\external\WaveTracer\external\limbo\tests\resources\digiTraceModels\9D_425)");
    Model m = Model::load(serialize::BinaryArchive(rootDir / "modelArchive"));

    serialize::exportFunction(
        rootDir / "export",
        serialize::GaussianProcess, // | serialize::LogLikelihood,
        m,
        8,
        [](std::string const& s) { spdlog::info(s); }
    );

    //m.optimize_hyperparams(); // TODO log each iteration
    // serialize::FunctionExport(
    //     rootDir / "post",
    //     serialize::FunctionExport::MeanFunction | serialize::FunctionExport::KernelFunction | serialize::FunctionExport::LogLikelihood,
    //     m,
    //     200);
    return 0;
}