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

#pragma once

#include <vector>
#include <map>
#include <string>

struct TensorStatistic {
    TensorStatistic(float* data, size_t count, size_t nbuckets = 1000);
    float getMaxValue() const;
    float getMinValue()const;
protected:
    float _min;
    float _max;
};

class AggregatedDataStats {
public:
    void addTensorStatistics(const std::string& name, size_t channel, float* data, size_t count);
    void addTensorStatistics(const std::string &name, size_t channel, uint8_t *data, size_t count);
    void getDataMinMax(const std::string& name, size_t channel, float& min, float& max, float threshold);
    size_t getNumberChannels(const std::string& name) const;
    std::vector <std::string> registeredLayers();
    void registerLayer(std::string layer);
protected:
    std::map<std::string, std::map<size_t, std::vector<TensorStatistic> > > _data;
};

