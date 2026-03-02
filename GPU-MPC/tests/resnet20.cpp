#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include <sytorch/module.h>
#include <sytorch/utils.h>

#include "backend/orca.h"
#include "utils/gpu_data_types.h"
#include "utils/gpu_file_utils.h"
#include "utils/gpu_mem.h"

using T = u64;

template <typename U>
class ResNet20Cifar : public SytorchModule<U> {
    using SytorchModule<U>::add;

    struct BasicBlock {
        Conv2D<U> *conv1;
        ReLU<U> *relu1;
        Conv2D<U> *conv2;
        Conv2D<U> *proj;
        ReLU<U> *relu_out;
        bool has_proj;
    };

    Conv2D<U> *stem_conv;
    ReLU<U> *stem_relu;
    std::vector<BasicBlock> blocks;
    GlobalAvgPool2D<U> *gap;
    Flatten<U> *flatten;
    FC<U> *fc;

public:
    ResNet20Cifar() {
        stem_conv = new Conv2D<U>(3, 16, 3, 1, 1, true);
        stem_relu = new ReLU<U>();

        blocks.reserve(9);
        int in_channels = 16;
        const std::vector<int> out_channels = {16, 16, 16, 32, 32, 32, 64, 64, 64};
        const std::vector<int> strides = {1, 1, 1, 2, 1, 1, 2, 1, 1};

        for (size_t i = 0; i < out_channels.size(); ++i) {
            const int out_ch = out_channels[i];
            const int stride = strides[i];
            const bool need_proj = (stride != 1) || (in_channels != out_ch);
            blocks.push_back({
                new Conv2D<U>(in_channels, out_ch, 3, 1, stride, true),
                new ReLU<U>(),
                new Conv2D<U>(out_ch, out_ch, 3, 1, 1, true),
                need_proj ? new Conv2D<U>(in_channels, out_ch, 1, 0, stride, true) : nullptr,
                new ReLU<U>(),
                need_proj
            });
            in_channels = out_ch;
        }

        gap = new GlobalAvgPool2D<U>();
        flatten = new Flatten<U>();
        fc = new FC<U>(64, 10, true);
    }

    Tensor<U> &_forward(Tensor<U> &input) override {
        auto &x0 = stem_conv->forward(input);
        Tensor<U> *x = &stem_relu->forward(x0);

        for (auto &blk : blocks) {
            auto &y1 = blk.conv1->forward(*x);
            auto &y2 = blk.relu1->forward(y1);
            auto &y3 = blk.conv2->forward(y2);
            Tensor<U> &skip = blk.has_proj ? blk.proj->forward(*x) : *x;
            auto &sum = add(y3, skip);
            x = &blk.relu_out->forward(sum);
        }

        auto &pooled = gap->forward(*x);
        auto &flat = flatten->forward(pooled);
        auto &out = fc->forward(flat);
        return out;
    }
};

static void fill_random_input(Tensor<T> &inp, u64 seed) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<u64> dist(0, 255);
    const int scale_shift = dcf::orca::global::scale - 8;
    for (u64 i = 0; i < inp.size(); ++i) {
        inp.data[i] = (dist(rng) << scale_shift);
    }
}

static Tensor<T> make_input(u64 batch, u64 seed) {
    Tensor<T> inp({batch, 32, 32, 3});
    fill_random_input(inp, seed);
    return inp;
}

static void run_keygen(int party, const std::string &key_dir, u64 seed, u64 batch) {
    sytorch_init();
    std::srand(static_cast<unsigned>(seed));

    auto model = std::make_unique<ResNet20Cifar<T>>();
    auto inp = make_input(batch, seed + 17);

    model->init(dcf::orca::global::scale, inp);

    auto keygen = std::make_unique<OrcaKeygen<T>>(
        party, dcf::orca::global::bw, dcf::orca::global::scale, key_dir + "/resnet20");
    model->setBackend(keygen.get());
    model->optimize();

    inp.d_data = (T *)moveToGPU((u8 *)inp.data, inp.size() * sizeof(T), (Stats *)NULL);
    auto &out = model->forward(inp);
    keygen->output(out);
    keygen->close();
    printf("[keygen] party=%d done\n", party);
}

static void run_eval(int party, const std::string &peer_ip, const std::string &key_dir, int iters, u64 seed, u64 batch) {
    sytorch_init();
    std::srand(static_cast<unsigned>(seed));

    auto model = std::make_unique<ResNet20Cifar<T>>();
    auto inp = make_input(batch, seed + 17);
    model->init(dcf::orca::global::scale, inp);

    auto online = std::make_unique<Orca<T>>(
        party, peer_ip, dcf::orca::global::bw, dcf::orca::global::scale, key_dir + "/resnet20");
    model->setBackend(online.get());
    model->optimize();

    lseek(online->fd, 0, SEEK_SET);
    readKey(online->fd, online->keySize, online->startPtr, NULL);

    std::vector<double> e2e_ms;
    e2e_ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        online->keyBuf = online->startPtr;
        online->s.reset();
        online->peer->sync();

        auto start = std::chrono::high_resolution_clock::now();
        inp.d_data = (T *)moveToGPU((u8 *)inp.data, inp.size() * sizeof(T), &(online->s));
        auto &out = model->forward(inp);
        online->output(out);
        auto end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        e2e_ms.push_back(ms);
        printf("[eval] party=%d iter=%d e2e_time_ms=%.3f\n", party, i + 1, ms);
    }

    const double avg = std::accumulate(e2e_ms.begin(), e2e_ms.end(), 0.0) / std::max(1, iters);
    printf("[eval] party=%d avg_e2e_time_ms=%.3f\n", party, avg);
    online->close();
}

static int spawn_eval_child(const char *binary, int party, const std::string &key_dir, int iters, u64 seed, u64 batch) {
    const std::string p = std::to_string(party);
    const std::string it = std::to_string(iters);
    const std::string sd = std::to_string(seed);
    const std::string bs = std::to_string(batch);
    execl(binary, binary, "eval", p.c_str(), "127.0.0.1", key_dir.c_str(), it.c_str(), sd.c_str(), bs.c_str(), (char *)NULL);
    perror("execl failed");
    return 127;
}

static int spawn_keygen_child(const char *binary, int party, const std::string &key_dir, u64 seed, u64 batch) {
    const std::string p = std::to_string(party);
    const std::string sd = std::to_string(seed);
    const std::string bs = std::to_string(batch);
    execl(binary, binary, "keygen", p.c_str(), key_dir.c_str(), sd.c_str(), bs.c_str(), (char *)NULL);
    perror("execl failed");
    return 127;
}

static int run_keygen_subprocess(const char *binary, int party, const std::string &key_dir, u64 seed, u64 batch) {
    pid_t pid = fork();
    if (pid == 0) return spawn_keygen_child(binary, party, key_dir, seed, batch);
    if (pid < 0) return 1;
    int st = 0;
    waitpid(pid, &st, 0);
    return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}

static int run_local_sim(const char *binary, const std::string &key_dir, int iters, u64 seed, u64 batch) {
    int rc = run_keygen_subprocess(binary, SERVER0, key_dir, seed, batch);
    if (rc != 0) return rc;
    rc = run_keygen_subprocess(binary, SERVER1, key_dir, seed, batch);
    if (rc != 0) return rc;

    pid_t p0 = fork();
    if (p0 == 0) return spawn_eval_child(binary, SERVER0, key_dir, iters, seed, batch);
    if (p0 < 0) return 1;

    sleep(1);
    pid_t p1 = fork();
    if (p1 == 0) return spawn_eval_child(binary, SERVER1, key_dir, iters, seed, batch);
    if (p1 < 0) return 1;

    int st0 = 0, st1 = 0;
    waitpid(p0, &st0, 0);
    waitpid(p1, &st1, 0);
    return (WIFEXITED(st0) ? WEXITSTATUS(st0) : 1) | (WIFEXITED(st1) ? WEXITSTATUS(st1) : 1);
}

static void usage(const char *argv0) {
    printf("Usage:\n");
    printf("  %s keygen <party:0|1> <key_dir> [seed=12345] [batch=1]\n", argv0);
    printf("  %s eval <party:0|1> <peer_ip> <key_dir> [iters=3] [seed=12345] [batch=1]\n", argv0);
    printf("  %s local-sim <key_dir> [iters=3] [seed=12345] [batch=1]\n", argv0);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const std::string mode = argv[1];
    if (mode == "keygen") {
        if (argc < 4) {
            usage(argv[0]);
            return 1;
        }
        const int party = std::atoi(argv[2]);
        const std::string key_dir = argv[3];
        const u64 seed = (argc > 4) ? std::strtoull(argv[4], nullptr, 10) : 12345ULL;
        const u64 batch = (argc > 5) ? std::strtoull(argv[5], nullptr, 10) : 1ULL;
        std::filesystem::create_directories(key_dir);
        run_keygen(party, key_dir, seed, batch);
        return 0;
    }

    if (mode == "eval") {
        if (argc < 5) {
            usage(argv[0]);
            return 1;
        }
        const int party = std::atoi(argv[2]);
        const std::string peer_ip = argv[3];
        const std::string key_dir = argv[4];
        const int iters = (argc > 5) ? std::atoi(argv[5]) : 3;
        const u64 seed = (argc > 6) ? std::strtoull(argv[6], nullptr, 10) : 12345ULL;
        const u64 batch = (argc > 7) ? std::strtoull(argv[7], nullptr, 10) : 1ULL;
        run_eval(party, peer_ip, key_dir, iters, seed, batch);
        return 0;
    }

    if (mode == "local-sim") {
        if (argc < 3) {
            usage(argv[0]);
            return 1;
        }
        const std::string key_dir = argv[2];
        const int iters = (argc > 3) ? std::atoi(argv[3]) : 3;
        const u64 seed = (argc > 4) ? std::strtoull(argv[4], nullptr, 10) : 12345ULL;
        const u64 batch = (argc > 5) ? std::strtoull(argv[5], nullptr, 10) : 1ULL;
        std::filesystem::create_directories(key_dir);
        return run_local_sim(argv[0], key_dir, iters, seed, batch);
    }

    usage(argv[0]);
    return 1;
}
