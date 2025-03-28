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
#ifndef LIMBO_SERIALIZE_TEXT_ARCHIVE_HPP
#define LIMBO_SERIALIZE_TEXT_ARCHIVE_HPP

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <fstream>

// Quick hack for definition of 'I' in <complex.h>
#undef I

#include <filesystem>
#include <Eigen/Core>

namespace limbo {
    namespace serialize {
	    namespace txt_impl
	    {

            inline std::string getFileName(std::filesystem::path const& dir, std::string const& object_name)
            {
                return (dir / (object_name + ".dat")).string();
            }

            inline std::vector<std::vector<double>> load(std::filesystem::path const& dir, std::string const& object_name)
            {
                std::ifstream ifs(getFileName(dir, object_name).c_str());
                assert(ifs.good() && "file not found");
                std::string line;
                std::vector<std::vector<double>> v;
                while (std::getline(ifs, line)) {
                    std::stringstream line_stream(line);
                    std::string cell;
                    std::vector<double> lineVec;
                    while (std::getline(line_stream, cell, ' '))
                        lineVec.push_back(std::stod(cell));
                    v.push_back(lineVec);
                }
                assert(!v.empty() && "empty file");
                return v;
            }
	    }

        class TextArchive {
        public:
            TextArchive(std::filesystem::path const& dir_name) : _dir_name(dir_name),
                                                       _fmt(Eigen::FullPrecision, Eigen::DontAlignCols, " ", "\n", "", "") {}

            /// write an Eigen::Matrix*
            void save(const Eigen::MatrixXd& v, const std::string& object_name) const
            {
                create_directories(_dir_name);
            	std::ofstream ofs(txt_impl::getFileName(_dir_name, object_name).c_str());
                ofs << v.format(_fmt) << std::endl;
            }

            /// write a vector of Eigen::Vector*
            template <typename T>
            void save(const std::vector<T>& v, const std::string& object_name) const
            {
                create_directories(_dir_name);
                std::ofstream ofs(txt_impl::getFileName(_dir_name, object_name).c_str());
                if constexpr (std::is_same_v<T, double>)
                {
                    ofs << std::setprecision(15);
		             for (auto const& x : v)
		             {
	                     ofs << x << "\n";
		             }
                }
                else
                {
                    for (auto& x : v) {
                        ofs << x.transpose().format(_fmt) << "\n";
                    }
                }
             
            }

            /// load an Eigen matrix (or vector)
            template <typename M>
            void load(M& m, const std::string& object_name) const
            {
                auto values = txt_impl::load(_dir_name, object_name);
                m.resize(values.size(), values[0].size());
                for (size_t i = 0; i < values.size(); ++i)
                    for (size_t j = 0; j < values[i].size(); ++j)
                        m(i, j) = values[i][j];
            }

            /// load a vector of Eigen::Vector*
            template <typename V>
            void load(std::vector<V>& m_list, const std::string& object_name) const
            {
                m_list.clear();
                auto values = txt_impl::load(_dir_name, object_name);
                assert(!values.empty());
                for (size_t i = 0; i < values.size(); ++i) {
                    if constexpr (std::is_same_v<double, V>)
                    {
                        m_list.push_back(values[i][0]);
                    }
                    else
                    {
                        V v(values[i].size());
                        for (size_t j = 0; j < values[i].size(); ++j)
                            v(j) = values[i][j];
                        m_list.push_back(v);
                    }
                }
                assert(!m_list.empty());
            }

        protected:
            std::filesystem::path _dir_name;
            Eigen::IOFormat _fmt;
        };
    } // namespace serialize
} // namespace limbo

#endif