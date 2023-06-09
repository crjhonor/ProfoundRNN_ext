"""
I am attempting to create a huge model, similar to GPT model which consist of millions of parameters and can be trained
with varied features without the limitation out of time windows and different targets.

And, before the construction of the model, hyperparameters have to be decided.
"""

VOCAB_SIZE = 10000 # This should be better than 1000 because it will provide more insignt into the feature
MAX_LENGTH = 128 # For multi gets dataset, the lengths will different quite sharply, thus using a large max_length
NUM_ATTENTION_HEADS = 12 # standard design
NUM_HIDDEN_LAYERS = 6 # 6 layers should be ok

"""
Dataset generations are multi gets. Not only trading data of single target is considered, but also multi targets to one
using sequence to sequence technic are used to generate the final dataset.
"""

