// Standalone latency benchmark for the UMT policy forward pass.
// Times isaaclab::OrtRunner::act() — the exact call State_UmtMimic makes each
// tick — with zero-filled observations. No DDS, robot, or joystick needed.
//
//   ./bench_umt [policy.onnx] [iterations]
//
// Defaults assume it is run from the build/ directory.

#include "isaaclab/algorithms/algorithms.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

int main(int argc, char** argv)
{
    std::string model_path = argc > 1 ? argv[1]
        : "../config/policy/umt/v0/exported/policy.onnx";
    const int iters = argc > 2 ? std::stoi(argv[2]) : 2000;
    const int warmup = 100;

    printf("model: %s\n", model_path.c_str());
    isaaclab::OrtRunner runner(model_path);

    std::unordered_map<std::string, std::vector<float>> obs;
    const auto& names = runner.get_input_names();
    const auto& sizes = runner.get_input_sizes();
    for (size_t i = 0; i < names.size(); ++i) {
        printf("input %s: %ld floats\n", names[i], (long)sizes[i]);
        obs[names[i]] = std::vector<float>(sizes[i], 0.0f);
    }

    for (int i = 0; i < warmup; ++i) runner.act(obs);

    std::vector<double> us(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::steady_clock::now();
        runner.act(obs);
        auto t1 = std::chrono::steady_clock::now();
        us[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }

    std::sort(us.begin(), us.end());
    double sum = 0;
    for (double v : us) sum += v;
    auto pct = [&](double p) { return us[std::min((size_t)(p * iters), us.size() - 1)]; };

    printf("\n%d iterations (after %d warmup):\n", iters, warmup);
    printf("  mean %.1f us | p50 %.1f | p90 %.1f | p99 %.1f | max %.1f\n",
           sum / iters, pct(0.50), pct(0.90), pct(0.99), us.back());
    printf("  (50 Hz budget = 20000 us, 500 Hz budget = 2000 us)\n");
    return 0;
}
