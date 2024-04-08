using Flux
# I don't have much evidence that NOM will be better than a gradient-free method
# (or perhaps the NOM is able to use a gradient-descent method because we have
# a differentiable model, which could be a plus), but we only have to beat random.
# https://arxiv.org/abs/2208.03897
