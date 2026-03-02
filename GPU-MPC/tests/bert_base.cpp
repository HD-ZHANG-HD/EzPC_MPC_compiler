#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <numeric>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include <llama/api.h>
#include <llama/comms.h>
#include <sytorch/backend/llama_transformer.h>
#include <sytorch/layers/layers.h>
#include <sytorch/module.h>
#include <sytorch/utils.h>

using T = u64;

template <typename U>
class BertFFN : public SytorchModule<U> {
    using SytorchModule<U>::gelu;

public:
    FC<U> *up;
    FC<U> *down;

    BertFFN(u64 in, u64 hidden) {
        up = new FC<U>(in, hidden, true);
        down = new FC<U>(hidden, in, true);
    }

    Tensor<U> &_forward(Tensor<U> &input) override {
        return down->forward(gelu(up->forward(input)));
    }
};

template <typename U>
class BertMHA : public SytorchModule<U> {
    using SytorchModule<U>::split;
    using SytorchModule<U>::view;
    using SytorchModule<U>::transpose;
    using SytorchModule<U>::matmul;
    using SytorchModule<U>::scalarmul;
    using SytorchModule<U>::softmax;
    using SytorchModule<U>::concat;

public:
    FC<U> *qkv;
    FC<U> *proj;
    u64 n_heads;
    u64 n_embd;

    BertMHA(u64 heads, u64 embd) : n_heads(heads), n_embd(embd) {
        always_assert(n_embd % n_heads == 0);
        qkv = new FC<U>(n_embd, 3 * n_embd, true);
        proj = new FC<U>(n_embd, n_embd, true);
    }

    Tensor<U> &_forward(Tensor<U> &input) override {
        auto &x = qkv->forward(input);
        auto &qkv_heads = split(x, 3);
        auto &q_heads = view(qkv_heads, 0);
        auto &k_heads = view(qkv_heads, 1);
        auto &v_heads = view(qkv_heads, 2);
        auto &qs = split(q_heads, n_heads);
        auto &ks = split(k_heads, n_heads);
        auto &vs = split(v_heads, n_heads);

        const double divisor = 1.0 / std::sqrt(double(n_embd) / double(n_heads));
        std::vector<Tensor<U> *> outputs;
        outputs.reserve(n_heads);

        for (u64 i = 0; i < n_heads; ++i) {
            auto &q = view(qs, i);
            auto &k = view(ks, i);
            auto &v = view(vs, i);
            auto &kt = transpose(k);
            auto &qk = matmul(q, kt);
            auto &scaled = scalarmul(qk, divisor);
            auto &probs = softmax(scaled);
            auto &head_out = matmul(probs, v);
            outputs.push_back(&head_out);
        }

        auto &cat = concat(outputs);
        return proj->forward(cat);
    }
};

template <typename U>
class BertBlock : public SytorchModule<U> {
    using SytorchModule<U>::add;

public:
    BertMHA<U> *attn;
    BertFFN<U> *ffn;
    LayerNorm<U> *ln0;
    LayerNorm<U> *ln1;

    BertBlock(u64 n_heads, u64 n_embd) {
        attn = new BertMHA<U>(n_heads, n_embd);
        ffn = new BertFFN<U>(n_embd, 4 * n_embd);
        ln0 = new LayerNorm<U>(n_embd);
        ln1 = new LayerNorm<U>(n_embd);
    }

    Tensor<U> &_forward(Tensor<U> &input) override {
        auto &attn_out = attn->forward(input);
        auto &add0 = add(attn_out, input);
        auto &norm0 = ln0->forward(add0);
        auto &ffn_out = ffn->forward(norm0);
        auto &add1 = add(ffn_out, norm0);
        return ln1->forward(add1);
    }
};

template <typename U>
class BertBaseModel : public SytorchModule<U> {
public:
    std::vector<BertBlock<U> *> blocks;
    LayerNorm<U> *final_ln;
    u64 n_layer;

    BertBaseModel(u64 layers, u64 heads, u64 embd) : n_layer(layers) {
        for (u64 i = 0; i < n_layer; ++i) {
            blocks.push_back(new BertBlock<U>(heads, embd));
        }
        final_ln = new LayerNorm<U>(embd);
    }

    Tensor<U> &_forward(Tensor<U> &input) override {
        Tensor<U> *x = &input;
        for (u64 i = 0; i < n_layer; ++i) {
            x = &(blocks[i]->forward(*x));
        }
        return final_ln->forward(*x);
    }
};

static void fill_random_input(Tensor<T> &inp, u64 seed, u64 scale) {
    std::srand((unsigned)seed);
    for (u64 i = 0; i < inp.size(); ++i) {
        inp.data[i] = (T(std::rand()) & ((1ULL << 20) - 1)) << (scale > 10 ? (scale - 10) : 0);
    }
}

template <typename U>
static void randomize_model_weights(SytorchModule<U> &model, u64 seed, u64 scale) {
    std::srand((unsigned)seed);
    for (auto *node : model.allNodesInExecutionOrder) {
        auto w = node->layer->getweights();
        auto b = node->layer->getbias();
        for (u64 i = 0; i < w.size; ++i) {
            w.data[i] = (U(std::rand()) & ((1ULL << 18) - 1)) << (scale > 10 ? (scale - 10) : 0);
        }
        for (u64 i = 0; i < b.size; ++i) {
            b.data[i] = (U(std::rand()) & ((1ULL << 18) - 1)) << (scale > 10 ? (scale - 10) : 0);
        }
    }
}

static void init_llama_config(int party, int bitlength) {
    LlamaConfig::party = party;
    LlamaConfig::bitlength = bitlength;
    LlamaConfig::stochasticT = false;
    LlamaConfig::stochasticRT = false;
    LlamaConfig::num_threads = 4;
}

static int run_dealer(const std::string &key_dir, u64 seed, u64 seq_len) {
    sytorch_init();
    std::filesystem::create_directories(key_dir);
    std::filesystem::current_path(key_dir);

    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;
    const u64 scale = 12;
    const int bitlength = 50;

    init_llama_config(DEALER, bitlength);
    auto llama = std::make_unique<LlamaTransformer<T>>();
    llama->init("127.0.0.1", true);

    BertBaseModel<T> model(n_layer, n_head, n_embd);
    Tensor<T> input({seq_len, n_embd});
    fill_random_input(input, seed, scale);

    model.init(scale, input);
    model.zero();
    model.setBackend(llama.get());
    model.optimize();

    llama->initializeInferencePartyA(model.root);
    llama->initializeInferencePartyB(input);

    auto t0 = std::chrono::high_resolution_clock::now();
    llama::start();
    auto &out = model.forward(input);
    llama::end();
    llama->outputA(out);
    auto t1 = std::chrono::high_resolution_clock::now();
    llama->finalize();

    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    printf("[dealer] generated keys, setup_time_ms=%.3f\n", ms);
    return 0;
}

static int run_eval_party(int party, const std::string &peer_ip, const std::string &key_dir, u64 seed, u64 seq_len) {
    sytorch_init();
    std::filesystem::current_path(key_dir);

    const u64 n_embd = 768;
    const u64 n_head = 12;
    const u64 n_layer = 12;
    const u64 scale = 12;
    const int bitlength = 50;

    init_llama_config(party, bitlength);
    auto llama = std::make_unique<LlamaTransformer<T>>();
    llama->init(peer_ip, true);

    BertBaseModel<T> model(n_layer, n_head, n_embd);
    Tensor<T> input({seq_len, n_embd});

    model.init(scale, input);
    if (party == SERVER) {
        randomize_model_weights(model, seed, scale);
        input.zero();
    } else {
        model.zero();
        fill_random_input(input, seed, scale);
    }
    model.setBackend(llama.get());
    model.optimize();

    llama->initializeInferencePartyA(model.root);
    llama->initializeInferencePartyB(input);

    auto t0 = std::chrono::high_resolution_clock::now();
    llama::start();
    auto &out = model.forward(input);
    llama::end();
    llama->outputA(out);
    auto t1 = std::chrono::high_resolution_clock::now();
    llama->finalize();

    double ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    printf("[eval] party=%d e2e_time_ms=%.3f\n", party, ms);
    return 0;
}

static int spawn_eval_child(const char *binary, int party, const std::string &key_dir, u64 seed, u64 seq_len) {
    const std::string p = std::to_string(party);
    const std::string sd = std::to_string(seed);
    const std::string sl = std::to_string(seq_len);
    execl(binary, binary, "eval", p.c_str(), "127.0.0.1", key_dir.c_str(), sd.c_str(), sl.c_str(), (char *)NULL);
    perror("execl failed");
    return 127;
}

static int spawn_dealer_child(const char *binary, const std::string &key_dir, u64 seed, u64 seq_len) {
    const std::string sd = std::to_string(seed);
    const std::string sl = std::to_string(seq_len);
    execl(binary, binary, "dealer", key_dir.c_str(), sd.c_str(), sl.c_str(), (char *)NULL);
    perror("execl failed");
    return 127;
}

static int run_local_sim(const char *binary, const std::string &key_dir, u64 seed, u64 seq_len) {
    pid_t dealer = fork();
    if (dealer == 0) return spawn_dealer_child(binary, key_dir, seed, seq_len);
    if (dealer < 0) return 1;
    int dealer_st = 0;
    waitpid(dealer, &dealer_st, 0);
    if (!(WIFEXITED(dealer_st) && WEXITSTATUS(dealer_st) == 0)) return 1;

    pid_t server = fork();
    if (server == 0) return spawn_eval_child(binary, SERVER, key_dir, seed, seq_len);
    if (server < 0) return 1;

    sleep(1);
    pid_t client = fork();
    if (client == 0) return spawn_eval_child(binary, CLIENT, key_dir, seed, seq_len);
    if (client < 0) return 1;

    int ss = 0, cs = 0;
    waitpid(server, &ss, 0);
    waitpid(client, &cs, 0);
    return (WIFEXITED(ss) ? WEXITSTATUS(ss) : 1) | (WIFEXITED(cs) ? WEXITSTATUS(cs) : 1);
}

static void usage(const char *argv0) {
    printf("Usage:\n");
    printf("  %s dealer <key_dir> [seed=12345] [seq_len=8]\n", argv0);
    printf("  %s eval <party:2|3> <peer_ip> <key_dir> [seed=12345] [seq_len=8]\n", argv0);
    printf("  %s local-sim <key_dir> [seed=12345] [seq_len=8]\n", argv0);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const std::string mode = argv[1];
    if (mode == "dealer") {
        if (argc < 3) {
            usage(argv[0]);
            return 1;
        }
        const std::string key_dir = argv[2];
        const u64 seed = (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 12345ULL;
        const u64 seq_len = (argc > 4) ? std::strtoull(argv[4], nullptr, 10) : 8ULL;
        return run_dealer(key_dir, seed, seq_len);
    }

    if (mode == "eval") {
        if (argc < 5) {
            usage(argv[0]);
            return 1;
        }
        const int party = std::atoi(argv[2]);
        const std::string peer_ip = argv[3];
        const std::string key_dir = argv[4];
        const u64 seed = (argc > 5) ? std::strtoull(argv[5], nullptr, 10) : 12345ULL;
        const u64 seq_len = (argc > 6) ? std::strtoull(argv[6], nullptr, 10) : 8ULL;
        if (party != SERVER && party != CLIENT) {
            fprintf(stderr, "eval party must be SERVER(2) or CLIENT(3)\n");
            return 1;
        }
        return run_eval_party(party, peer_ip, key_dir, seed, seq_len);
    }

    if (mode == "local-sim") {
        if (argc < 3) {
            usage(argv[0]);
            return 1;
        }
        const std::string key_dir = argv[2];
        const u64 seed = (argc > 3) ? std::strtoull(argv[3], nullptr, 10) : 12345ULL;
        const u64 seq_len = (argc > 4) ? std::strtoull(argv[4], nullptr, 10) : 8ULL;
        return run_local_sim(argv[0], key_dir, seed, seq_len);
    }

    usage(argv[0]);
    return 1;
}
