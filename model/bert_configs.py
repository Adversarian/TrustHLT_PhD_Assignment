target = {
    "hidden_size": 512,
    "hidden_act": "gelu",
    "initializer_range": 0.02,
    "vocab_size": 30522,
    "hidden_dropout_prob": 0.1,
    "num_attention_heads": 8,
    "type_vocab_size": 2,
    "max_position_embeddings": 512,
    "num_hidden_layers": 4,
    "intermediate_size": 2048,
    "attention_probs_dropout_prob": 0.1,
    "num_labels": 5,
}


shadow = {
    "hidden_size": 256,
    "hidden_act": "gelu",
    "initializer_range": 0.02,
    "vocab_size": 30522,
    "hidden_dropout_prob": 0.1,
    "num_attention_heads": 4,
    "type_vocab_size": 2,
    "max_position_embeddings": 512,
    "num_hidden_layers": 4,
    "intermediate_size": 1024,
    "attention_probs_dropout_prob": 0.1,
    "num_labels": 5,
}
