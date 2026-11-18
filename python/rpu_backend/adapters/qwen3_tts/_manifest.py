"""Frozen public checkpoint and reviewed source identities; no executable upstream code."""

CHECKPOINT_REVISION = "5d83992436eae1d760afd27aff78a71d676296fc"

SOURCE_REVISION = "022e286b98fbec7e1e916cb940cdf532cd9f488e"

SOURCE_HASHES = {
    "__init__.py": "ea52de59d070fde366467a6902d0edcfc1b0575b8c570a0c71020c41d6a593ed",
    "core/__init__.py": "1b380d9de843b6d585d938c339d066136567ca7125412674234204af4386679e",
    "core/models/__init__.py": "917d38d8fd7ed84de73703fd2394d535ca416e15694d1e50324e4a06d4022c82",
    "core/models/configuration_qwen3_tts.py": "f52867f14fde06a416dd14864d503ce6d13d0a08d5f5da30191e1c80c13f5d18",
    "core/models/modeling_qwen3_tts.py": "25c42656bcf810f06ef6bc1839bd7083f3c8cfedac3a147c4060b4262b1c96a0",
    "core/models/processing_qwen3_tts.py": "61472203188e46eda68d740c25f0f228ece24a61ecf942e57d6e49e78b4d6bcd",
    "core/tokenizer_12hz/configuration_qwen3_tts_tokenizer_v2.py": "9e30c24394b00cb0366d7da3482b7436468acc1cd3da1a6fe614a1d34653a5e3",
    "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py": "844e8dd8c0182ef9c6463c874631c22ef3c5a4fd1899dd657016164cc5379628",
    "core/tokenizer_25hz/configuration_qwen3_tts_tokenizer_v1.py": "28f94f715df7d96d365c9cea0021d9cca8c63d2f48f48423c43a3d3cae28ef02",
    "core/tokenizer_25hz/modeling_qwen3_tts_tokenizer_v1.py": "8c6d5b7cfb133967f04286e069138ef7fe0cfca30e9acf3326106f6117135a7f",
    "core/tokenizer_25hz/vq/core_vq.py": "d35fa9f00ef4863361493e102f4e28f092488497117df580819e93b5dff3b8de",
    "core/tokenizer_25hz/vq/speech_vq.py": "7c68fcba8c508d8a18e2241b563302d1bf46f8ecb1515c59222263d8524bb298",
    "core/tokenizer_25hz/vq/whisper_encoder.py": "a1748b34968ba8effebf902c4a4399941b3e49bca3f6c0120fc673e3ba7d767d",
    "inference/qwen3_tts_model.py": "e1da450732857c1f5fe3e36ebab85db2f6dc6a48caaff6973f463384e30275e4",
    "inference/qwen3_tts_tokenizer.py": "ac2d855022a1bd21d33ab7b267ec952eef71f81d3f8d969a306139ff1a929515"
}

PREPARED_HASHES = {
    "__init__.py": "ea52de59d070fde366467a6902d0edcfc1b0575b8c570a0c71020c41d6a593ed",
    "core/__init__.py": "1b380d9de843b6d585d938c339d066136567ca7125412674234204af4386679e",
    "core/models/__init__.py": "917d38d8fd7ed84de73703fd2394d535ca416e15694d1e50324e4a06d4022c82",
    "core/models/configuration_qwen3_tts.py": "f52867f14fde06a416dd14864d503ce6d13d0a08d5f5da30191e1c80c13f5d18",
    "core/models/modeling_qwen3_tts.py": "9d30973e6827f1d377d1d1c4e22081c75fa4c9ba9009cd3c159735b653979f2d",
    "core/models/processing_qwen3_tts.py": "61472203188e46eda68d740c25f0f228ece24a61ecf942e57d6e49e78b4d6bcd",
    "core/tokenizer_12hz/configuration_qwen3_tts_tokenizer_v2.py": "9e30c24394b00cb0366d7da3482b7436468acc1cd3da1a6fe614a1d34653a5e3",
    "core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py": "458ccbe535e8cc748872abe3f773d27631ba3de099b5994699e448d80a679614",
    "core/tokenizer_25hz/configuration_qwen3_tts_tokenizer_v1.py": "28f94f715df7d96d365c9cea0021d9cca8c63d2f48f48423c43a3d3cae28ef02",
    "core/tokenizer_25hz/modeling_qwen3_tts_tokenizer_v1.py": "8c6d5b7cfb133967f04286e069138ef7fe0cfca30e9acf3326106f6117135a7f",
    "core/tokenizer_25hz/vq/core_vq.py": "d35fa9f00ef4863361493e102f4e28f092488497117df580819e93b5dff3b8de",
    "core/tokenizer_25hz/vq/speech_vq.py": "7c68fcba8c508d8a18e2241b563302d1bf46f8ecb1515c59222263d8524bb298",
    "core/tokenizer_25hz/vq/whisper_encoder.py": "a1748b34968ba8effebf902c4a4399941b3e49bca3f6c0120fc673e3ba7d767d",
    "inference/qwen3_tts_model.py": "e1da450732857c1f5fe3e36ebab85db2f6dc6a48caaff6973f463384e30275e4",
    "inference/qwen3_tts_tokenizer.py": "ac2d855022a1bd21d33ab7b267ec952eef71f81d3f8d969a306139ff1a929515"
}

CHECKPOINT_HASHES = {
    "config.json": "2e714c787c8edb98b05432685cddb634add2de4d4e645f653d68251ef72ba011",
    "generation_config.json": "f1b90b4513f3b34c62851049e2492d7b4c5940daf1276f89c82b8ef04127f3aa",
    "tokenizer_config.json": "dc3c31c3bdaedd5016382bb3cbe07323026775ad51f5a4fb564505992ae4a670",
    "preprocessor_config.json": "efdde1022ea9d76928bf7a9cd53139138f5ba2e466e837f08f6105ab1af1c119",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
    "speech_tokenizer/config.json": "ee65bb901c876664ab8707c487157aa1a6ee57c65969b28fb5ec9dc211e68167",
    "speech_tokenizer/preprocessor_config.json": "fcb3805e597e786d4067706e602f6688524640f8d3396790e2e09b5942fcbdfb",
    "model.safetensors": "180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6",
    "speech_tokenizer/model.safetensors": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258"
}

CHECKPOINT_PROFILES = {
    "0.6b": {
        "revision": CHECKPOINT_REVISION,
        "hashes": CHECKPOINT_HASHES,
        "predictor_mlp_up_divisors": {2: 8},
    },
    "1.7b": {
        "revision": "fd4b254389122332181a7c3db7f27e918eec64e3",
        "predictor_mlp_up_divisors": {},
        "hashes": {
            **CHECKPOINT_HASHES,
            "config.json": "b4f01752d15a488abde3e1ab44723ae4f4b9e68a4037257b098b3737893cc1f9",
            "model.safetensors": "38fc7fc51c5e776e840414b6fd443962e9411b9654888fd7913e4da643cb857c",
        },
    },
}
