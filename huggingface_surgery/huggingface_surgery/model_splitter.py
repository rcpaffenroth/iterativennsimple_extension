from collections import OrderedDict

def model_splitter(model, name, wrapper=None):
    model_parts = OrderedDict()
    if name == "tiiuae/falcon-rw-1b":
        # Add the embedding layer.
        if wrapper is not None:
            model_parts['embedding'] = wrapper(model.transformer.word_embeddings)
            model.transformer.word_embeddings = model_parts['embedding']
        else:
            model_parts['embedding'] = model.transformer.word_embeddings

        # Add each of the transformer layers.
        for i, layer in enumerate(model.transformer.h):
            if wrapper is not None:
                model_parts[f'transformer_layer_{i}'] = wrapper(layer)
                model.transformer.h[i] = model_parts[f'transformer_layer_{i}']
            else:
                model_parts[f'transformer_layer_{i}'] = layer

        # Add final layer norm.
        if wrapper is not None:
            model_parts['layer_norm'] = wrapper(model.transformer.ln_f)
            model.transformer.ln_f = model_parts['layer_norm']
        else:
            model_parts['layer_norm'] = model.transformer.ln_f

        # Add the output layer.
        if wrapper is not None:
            model_parts['output'] = wrapper(model.lm_head)
            model.lm_head = model_parts['output']
        else:
            model_parts['output'] = model.lm_head
    else:
        raise ValueError(f'Unknown model name {name}')
    return model_parts