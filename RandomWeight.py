import numpy as np

class RandomQwen3Weights:
    def __init__(self, cfg, seed=0):
        self.rng = np.random.default_rng(seed)
        self.num_layers = cfg.num_layers
        self.eps = cfg.eps
        self.WE = self.rand(cfg.vocab_size, cfg.D)
        self.final_gamma = np.ones(cfg.D)
        self.output_matrix = self.rand(cfg.D, cfg.vocab_size)
        self.decoder_layer_parms = [self.make_layer(cfg) for _ in range(cfg.num_layers)]

    def rand(self, *shape):
        return self.rng.normal(0, 0.02, shape)


    def make_layer(self, cfg):
        return (
            np.ones(cfg.D),
            np.ones(cfg.D),
            cfg.eps,
            self.rand(cfg.D, cfg.H_q * cfg.d_h),
            self.rand(cfg.D, cfg.H_kv * cfg.d_h),
            self.rand(cfg.D, cfg.H_kv * cfg.d_h),
            self.rand(cfg.H_q * cfg.d_h, cfg.D),
            cfg.d_h,
            np.ones(cfg.d_h),
            np.ones(cfg.d_h),
            cfg.base,
            self.rand(cfg.D, cfg.I),
            self.rand(cfg.D, cfg.I),
            self.rand(cfg.I, cfg.D),
        )


# cfg = Qwen3Config()
# w = RandomQwen3Weights(cfg)
# model = Qwen3(w.num_layers, w.WE, w.final_gamma, w.eps, w.output_matrix, w.decoder_layer_parms)

# logits = model(np.array([[1, 2, 3, 4]])) 
# print(logits.shape) # (1, 4, 151936)