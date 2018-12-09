/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

/**
* @brief The entry point the Inference Engine sample application
* @file classification_sample/main.cpp
* @example classification_sample/main.cpp
*/

#include <fstream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <memory>
#include <string>
#include <map>

#include <inference_engine.hpp>

#include <format_reader_ptr.h>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>

#include <sys/stat.h>
#include <ext_list.hpp>

#include "classification_sample_async.h"

using namespace InferenceEngine;

ConsoleErrorListener error_listener;

bool ParseAndCheckCommandLine(int argc, char *argv[]) {
    // ---------------------------Parsing and validation of input args--------------------------------------
    slog::info << "Parsing input parameters" << slog::endl;

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        return false;
    }
    slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_ni < 1) {
        throw std::logic_error("Parameter -ni must be more than 0 ! (default 1)");
    }

    if (FLAGS_nireq < 1) {
        throw std::logic_error("Parameter -nireq must be more than 0 ! (default 1)");
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_m.empty()) {
        throw std::logic_error("Parameter -m is not set");
    }

    if (FLAGS_ni < FLAGS_nireq) {
        throw std::logic_error("Number of iterations could not be less than requests quantity");
    }

    return true;
}

int main(int argc, char *argv[]) {
    try {
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        // ------------------------------ Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        /** This vector stores paths to the processed images **/
        std::vector<std::string> imageNames;
        parseInputFilesArguments(imageNames);
        if (imageNames.empty()) throw std::logic_error("No suitable images were found");
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 1. Load Plugin for inference engine -------------------------------------
        slog::info << "Loading plugin" << slog::endl;
        InferencePlugin plugin = PluginDispatcher({ FLAGS_pp, "../../../lib/intel64" , "" }).getPluginByDevice(FLAGS_d);
        if (FLAGS_p_msg) {
            static_cast<InferenceEngine::InferenceEnginePluginPtr>(plugin)->SetLogCallback(error_listener);
        }

        /** Loading default extensions **/
        if (FLAGS_d.find("CPU") != std::string::npos) {
            /**
             * cpu_extensions library is compiled from "extension" folder containing
             * custom MKLDNNPlugin layer implementations. These layers are not supported
             * by mkldnn, but they can be useful for inferring custom topologies.
            **/
            plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
        }

        if (!FLAGS_l.empty()) {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
            plugin.AddExtension(extension_ptr);
            slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        }
        if (!FLAGS_c.empty()) {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            plugin.SetConfig({{PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c}});
            slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
        }

        ResponseDesc resp;
        /** Printing plugin version **/
        printPluginVersion(plugin, std::cout);
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
        slog::info << "Loading network files" << slog::endl;

        CNNNetReader networkReader;
        /** Read network model **/
        networkReader.ReadNetwork(FLAGS_m);

        /** Extract model name and load weights **/
        std::string binFileName = fileNameNoExt(FLAGS_m) + ".bin";
        networkReader.ReadWeights(binFileName);

        CNNNetwork network = networkReader.getNetwork();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------

        // --------------------------- Prepare input blobs -----------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputInfo(network.getInputsInfo());
        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");

        auto inputInfoItem = *inputInfo.begin();

        /** Specifying the precision and layout of input data provided by the user.
         * This should be called before load of the network to the plugin **/
        inputInfoItem.second->setPrecision(Precision::U8);
        inputInfoItem.second->setLayout(Layout::NCHW);

        std::vector<std::shared_ptr<unsigned char>> imagesData;
        for (auto & i : imageNames) {
            FormatReader::ReaderPtr reader(i.c_str());
            if (reader.get() == nullptr) {
                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
                continue;
            }
            /** Store image data **/
            std::shared_ptr<unsigned char> data(
                    reader->getData(inputInfoItem.second->getTensorDesc().getDims()[3],
                                    inputInfoItem.second->getTensorDesc().getDims()[2]));
            if (data.get() != nullptr) {
                imagesData.push_back(data);
            }
        }
        if (imagesData.empty()) throw std::logic_error("Valid input images were not found!");

        /** Setting batch size using image count **/
        network.setBatchSize(imagesData.size());
        size_t batchSize = network.getBatchSize();
        slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;

        // ------------------------------ Prepare output blobs -------------------------------------------------
        slog::info << "Preparing output blobs" << slog::endl;

        OutputsDataMap outputInfo(network.getOutputsInfo());
        std::vector <Blob::Ptr> outputBlobs;
        for (size_t i = 0; i < FLAGS_nireq; i++) {
            auto outputBlob = make_shared_blob<PrecisionTrait<Precision::FP32>::value_type>(outputInfo.begin()->second->getTensorDesc());
            outputBlob->allocate();
            outputBlobs.push_back(outputBlob);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 4. Loading model to the plugin ------------------------------------------
        slog::info << "Loading model to the plugin" << slog::endl;

        std::map<std::string, std::string> config;
        if (FLAGS_pc) {
            config[PluginConfigParams::KEY_PERF_COUNT] = PluginConfigParams::YES;
        }
        ExecutableNetwork executable_network = plugin.LoadNetwork(network, {});
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 5. Create infer request -------------------------------------------------
        std::vector<InferRequest> inferRequests;
        for (size_t i = 0; i < FLAGS_nireq; i++) {
            InferRequest inferRequest = executable_network.CreateInferRequest();
            inferRequests.push_back(inferRequest);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 6. Prepare input --------------------------------------------------------
        BlobMap inputBlobs;
        for (auto & item : inputInfo) {
            auto input = make_shared_blob<PrecisionTrait<Precision::U8>::value_type>(item.second->getTensorDesc());
            input->allocate();
            inputBlobs[item.first] = input;

            auto dims = input->getTensorDesc().getDims();
            /** Fill input tensor with images. First b channel, then g and r channels **/
            size_t num_channels = dims[1];
            size_t image_size = dims[3] * dims[2];

            /** Iterate over all input images **/
            for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
                /** Iterate over all pixel in image (b,g,r) **/
                for (size_t pid = 0; pid < image_size; pid++) {
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < num_channels; ++ch) {
                        /**          [images stride + channels stride + pixel id ] all in bytes            **/
                        input->data()[image_id * image_size * num_channels + ch * image_size + pid] = imagesData.at(image_id).get()[pid*num_channels + ch];
                    }
                }
            }
        }

        for (size_t i = 0; i < FLAGS_nireq; i++) {
            inferRequests[i].SetBlob(inputBlobs.begin()->first, inputBlobs.begin()->second);
            inferRequests[i].SetBlob(outputInfo.begin()->first, outputBlobs[i]);
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 7. Do inference ---------------------------------------------------------
        slog::info << "Start inference (" << FLAGS_ni << " iterations)" << slog::endl;

        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        typedef std::chrono::duration<float> fsec;

        double total = 0.0;
        /** Start inference & calc performance **/
        auto t0 = Time::now();

        size_t currentInfer = 0;
        size_t prevInfer = (FLAGS_nireq > 1) ? 1 : 0;


        // warming up
        inferRequests[0].StartAsync();
        inferRequests[0].Wait(10000);

        for (int iter = 0; iter < FLAGS_ni + FLAGS_nireq; ++iter) {
            if (iter < FLAGS_ni) {
                inferRequests[currentInfer].StartAsync();
            }
            inferRequests[prevInfer].Wait(10000);

            currentInfer++;
            if (currentInfer >= FLAGS_nireq) {
                currentInfer = 0;
            }
            prevInfer++;
            if (prevInfer >= FLAGS_nireq) {
                prevInfer = 0;
            }
        }
        auto t1 = Time::now();
        fsec fs = t1 - t0;
        ms d = std::chrono::duration_cast<ms>(fs);
        total = d.count();
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 8. Process output -------------------------------------------------------
        slog::info << "Processing output blobs" << slog::endl;

        for (size_t i = 0; i < FLAGS_nireq; i++) {
            /** Validating -nt value **/
            const int resultsCnt = outputBlobs[i]->size() / batchSize;
            if (FLAGS_nt > resultsCnt || FLAGS_nt < 1) {
                slog::warn << "-nt " << FLAGS_nt << " is not available for this network (-nt should be less than " \
                          << resultsCnt+1 << " and more than 0)\n            will be used maximal value : " << resultsCnt << slog::endl;
                FLAGS_nt = resultsCnt;
            }
            /** This vector stores id's of top N results **/
            std::vector<unsigned> results;
            TopResults(FLAGS_nt, *outputBlobs[i], results);

            std::cout << std::endl << "Top " << FLAGS_nt << " results:" << std::endl << std::endl;

            /** Read labels from file (e.x. AlexNet.labels) **/
            bool labelsEnabled = false;
            std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
            std::vector<std::string> labels;

            std::ifstream inputFile;
            inputFile.open(labelFileName, std::ios::in);
            if (inputFile.is_open()) {
                std::string strLine;
                while (std::getline(inputFile, strLine)) {
                    trim(strLine);
                    labels.push_back(strLine);
                }
                labelsEnabled = true;
            }

            /** Print the result iterating over each batch **/
            for (int image_id = 0; image_id < batchSize; ++image_id) {
                std::cout << "Image " << imageNames[image_id] << std::endl << std::endl;
                for (size_t id = image_id * FLAGS_nt, cnt = 0; cnt < FLAGS_nt; ++cnt, ++id) {
                    std::cout.precision(7);
                    /** Getting probability for resulting class **/
                    auto result = outputBlobs[i]->buffer().
                            as<PrecisionTrait<Precision::FP32>::value_type*>()[results[id] + image_id*(outputBlobs[i]->size() / batchSize)];
                    std::cout << std::left << std::fixed << results[id] << " " << result;
                    if (labelsEnabled) {
                        std::cout << " label " << labels[results[id]] << std::endl;
                    } else {
                        std::cout << " label #" << results[id] << std::endl;
                    }
                }
                std::cout << std::endl;
            }
        }
        // -----------------------------------------------------------------------------------------------------
        std::cout << std::endl << "total inference time: " << total << std::endl;
        std::cout << std::endl << "Throughput: " << 1000 * static_cast<double>(FLAGS_ni) * batchSize / total << " FPS" << std::endl;
        std::cout << std::endl;

        /** Show performance results **/
        std::map<std::string, InferenceEngineProfileInfo> performanceMap;
        if (FLAGS_pc) {
            for (size_t nireq = 0; nireq < FLAGS_nireq; nireq++) {
                performanceMap = inferRequests[nireq].GetPerformanceCounts();
                printPerformanceCounts(performanceMap, std::cout);
            }
        }
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}
