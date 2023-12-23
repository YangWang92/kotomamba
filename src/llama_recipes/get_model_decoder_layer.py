from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer


def get_model_decoder_layer(model_name: str) -> type[LlamaDecoderLayer] | type[MistralDecoderLayer]:
    if "Llama" in model_name:
        return LlamaDecoderLayer
    elif "Mistral" in model_name:
        return MistralDecoderLayer
    elif "calm2-7b" in model_name:
        return LlamaDecoderLayer
    elif "stockmark-13b" in model_name:
        return LlamaDecoderLayer
    elif "ELYZA-japanese-Llama-2-7b" in model_name:
        return LlamaDecoderLayer
    elif "japanese-stablelm-base-ja_vocab-beta-7b" in model_name:
        return LlamaDecoderLayer
    elif "japanese-stablelm-base-beta" in model_name:
        return LlamaDecoderLayer
    else:
        raise NotImplementedError(f"{model_name}: this model decoder layer is not implemented.")
