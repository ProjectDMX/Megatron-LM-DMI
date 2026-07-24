# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np

try:
    import transformers

    HAVE_TRANSFORMERS = True
except ModuleNotFoundError:
    HAVE_TRANSFORMERS = False


# fmt: off
nemotron_h_aligned_custom_template = """{% for message in messages %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + message['content'].strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + message['content'].strip() + '\n' + '<SPECIAL_11>Assistant\n' }}{% elif message['role'] == 'assistant' %}{{ message['content'].strip() + '\n' }}{% endif %}{% endfor %}""" # pylint: disable=line-too-long
nemotron_nano_v2_custom_template = """{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'system' %}{{ '<SPECIAL_10>System\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'user' %}{{ '<SPECIAL_11>User\n' + content.replace('/think', '').replace('/no_think', '').strip() + '\n' }}{% elif message['role'] == 'assistant' %}{{ '<SPECIAL_11>Assistant\n' + content.strip() + '\n<SPECIAL_12>\n' }}{% endif %}{% endfor %}""" # pylint: disable=line-too-long
identity_template = """{% for message in messages %}{{ message['content'] }}{% endfor %}"""
# fmt: on


IGNORE_INDEX = -100


@dataclass
class PromptConfig:
    """Config options for different prompt formats."""

    # How many tokens are used for the assistant prefix, e.g. "<|im_start|>assistant\n".
    # Used for masking the assistant prefix.
    assistant_prefix_len: int
    # Padding token ID.
    pad_token_id: int
    # For overriding the default chat format template.
    custom_chat_template: str
    # If the tokenizer inserts BOS token by default.
    has_bos: bool
    # If the tokenizer supports a separate role for system messages.
    has_system_role: bool
    # Wether to force a specific system message.
    force_system_message: bool = False
    system_default: dict = None


class SFTTokenizer:
    """SFT Tokenizer."""

    def __init__(self, tokenizer_path: str, prompt_format: str):
        """
        Note: Currently, only HuggingFaceTokenizer is supported as the underlying text tokenizer.

        Args:
            tokenizer_path (str): Underlying tokenizer path.
            prompt_format (str): Prompt format for the tokenizer.
        """
        if HAVE_TRANSFORMERS:
            # Currently, only HuggingFace tokenizers are supported.
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=tokenizer_path
            )
        else:
            raise ImportError(
                "SFTTokenizer currently requires transformers library to be installed"
            )

        self._vocab_size = len(tokenizer)
        self._tokenizer = tokenizer

        if prompt_format == "nemotron-nano-v2":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=3,
                pad_token_id=tokenizer.convert_tokens_to_ids("<unk>"),
                custom_chat_template=nemotron_nano_v2_custom_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "nemotron-h-aligned":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.convert_tokens_to_ids("<SPECIAL_233>"),
                custom_chat_template=nemotron_h_aligned_custom_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "identity":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=tokenizer.convert_tokens_to_ids("<unk>"),
                custom_chat_template=identity_template,
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "vicuna-v1.1":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=(
                    tokenizer.unk_token_id
                    if tokenizer.unk_token_id is not None
                    else tokenizer.eos_token_id
                ),
                custom_chat_template="",
                has_bos=False,
                has_system_role=True,
            )
        elif prompt_format == "default":
            self._prompt_config = PromptConfig(
                assistant_prefix_len=0,
                pad_token_id=(
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
                custom_chat_template=tokenizer.chat_template,
                has_bos=tokenizer.bos_token_id is not None,
                has_system_role=True,
            )
        else:
            raise NotImplementedError("unknown SFT prompt format", prompt_format)

        self._prompt_format = prompt_format

    def tokenize_conversation(
        self, conversation: List[Dict], return_target: bool, add_generation_prompt: bool
    ):
        """Convert a conversation to tokens.

        Args:
            conversation (List[Dict]): Sequence of system/user/assistant messages.
                Must be in the following format:
                [
                    {"role": "system", "content": "something"},
                    {"role": "user", "content": "something1"},
                    {"role": "assistant", "content": "something2"},
                ]
            return_target (bool): Return target tokens with system and assistant masked.
            add_generation_prompt (bool): Add assistant prefix to the end.
        """
        if self._prompt_format == "vicuna-v1.1":
            return self._tokenize_vicuna_v1_1(
                conversation,
                return_target=return_target,
                add_generation_prompt=add_generation_prompt,
            )

        # Skip system message if the tokenizer doesn't have a system role.
        if not self._prompt_config.has_system_role and conversation[0]["role"] == "system":
            conversation = conversation[1:]

        tokens = self._tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
            return_assistant_token_mask=False,
            return_tensors="np",
            chat_template=self._prompt_config.custom_chat_template,
        )[0]

        if not return_target:
            return tokens

        target = tokens.copy()

        # When using the default prompt format, we do not replace any tokens with IGNORE_INDEX.
        # Instead, all token losses will be used for simplicity.
        if self._prompt_format == "default":
            return tokens, target

        # Mask system and user tokens in the target.
        idx = 0
        for turn_idx, turn in enumerate(conversation):

            if turn["role"].lower() == "assistant" and len(turn["content"]) == 0:
                raise ValueError(f"empty assistant turn in conversation: {conversation}.")
            if turn["role"].lower() == "assistant":
                assert conversation[turn_idx - 1]["role"].lower() in ("user", "tool")

            turn_tokens = self._tokenizer.apply_chat_template(
                [turn], tokenize=True, chat_template=self._prompt_config.custom_chat_template
            )

            # There should be only one BOS at the very beginning.
            # After the first turn, skip BOS token.
            if self._prompt_config.has_bos and turn_idx > 0:
                turn_tokens = turn_tokens[1:]
            turn_len = len(turn_tokens)

            role = turn["role"].lower()
            if role in ("system", "user", "tool"):
                target[idx : idx + turn_len] = IGNORE_INDEX
            elif role == "assistant":
                if self._prompt_config.assistant_prefix_len > 0:
                    target[idx : idx + self._prompt_config.assistant_prefix_len] = IGNORE_INDEX
            else:
                raise ValueError("Wrong role value.")

            assert np.allclose(
                tokens[idx : idx + turn_len], turn_tokens
            ), f"expected turn tokens to match tokens in conversation {conversation}"

            idx += turn_len

        assert idx == len(tokens), f"mismatch in target masking the conversation {conversation}"

        return tokens, target

    def _tokenize_vicuna_v1_1(
        self,
        conversation: List[Dict],
        *,
        return_target: bool,
        add_generation_prompt: bool,
    ):
        if not conversation or conversation[0]["role"].lower() != "system":
            raise ValueError("vicuna-v1.1 requires an explicit leading system message")

        text_parts = [str(conversation[0]["content"]), " "]
        target_ranges = []
        expected_role = "user"
        cursor = len("".join(text_parts))
        for turn in conversation[1:]:
            role = str(turn["role"]).lower()
            content = str(turn["content"])
            if role != expected_role:
                raise ValueError(
                    f"vicuna-v1.1 expected role {expected_role!r}, got {role!r}"
                )
            if role == "user":
                piece = f"USER: {content} "
                text_parts.append(piece)
                cursor += len(piece)
                expected_role = "assistant"
                continue

            prefix = "ASSISTANT:"
            text_parts.append(prefix)
            cursor += len(prefix)
            if content:
                target_piece = f" {content}{self._tokenizer.eos_token}"
                text_parts.append(target_piece)
                target_ranges.append((cursor, cursor + len(target_piece)))
                cursor += len(target_piece)
            expected_role = "user"

        if add_generation_prompt:
            if expected_role != "assistant":
                raise ValueError(
                    "vicuna-v1.1 generation prompt requires a trailing user turn"
                )
            text_parts.append("ASSISTANT:")
        elif expected_role != "user":
            raise ValueError("vicuna-v1.1 training conversation must end with assistant")

        rendered = "".join(text_parts)
        encoded = self._tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=return_target,
        )
        tokens = np.asarray(encoded["input_ids"], dtype=np.int64)
        if not return_target:
            return tokens

        target = np.full(tokens.shape, IGNORE_INDEX, dtype=np.int64)
        offsets = encoded.get("offset_mapping")
        if offsets is None:
            raise RuntimeError(
                "vicuna-v1.1 assistant masking requires a fast tokenizer "
                "with offset_mapping support"
            )
        for token_index, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            if any(
                start < target_end and end > target_start
                for target_start, target_end in target_ranges
            ):
                target[token_index] = tokens[token_index]
        return tokens, target

    def text_to_ids(self, text: Union[str, List[Dict]]):
        """Tokenize conversation or string input."""
        if isinstance(text, list):
            # This code path is used by the inference code currently.
            return self.tokenize_conversation(
                text, return_target=False, add_generation_prompt=True
            ).tolist()

        return self._tokenizer.encode(text)

    def tokens_to_ids(self, tokens: List[str]):
        """Convert tokens to IDs."""
        return self._tokenizer.convert_tokens_to_ids(tokens)

    def ids_to_text(self, tokens: List[int]):
        """Detokenize tokens."""
        return self._tokenizer.decode(tokens)

    def ids_to_tokens(self):
        """Converts ids to tokens."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    def text_to_tokens(self):
        """Converts text to tokens."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    def tokens_to_text(self):
        """Converts tokens to text."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    def get_special_tokens(self):
        """Get special tokens."""
        return self._tokenizer.get_added_vocab()

    def add_special_tokens(self):
        """Add special tokens."""
        raise NotImplementedError("This method is not supported for SFTTokenizer.")

    @property
    def pad_id(self):
        """Pad token ID."""
        return self._prompt_config.pad_token_id

    @property
    def bos_id(self):
        """Beginning of sequence token ID."""
        return self._tokenizer.bos_token_id

    @property
    def eod(self):
        """End of sentence token ID."""
        return self._tokenizer.eos_token_id

    @property
    def vocab(self):
        """Vocab."""
        return NotImplementedError("not used")

    @property
    def inv_vocab(self):
        """Inverse vocab."""
        return NotImplementedError("not used")

    @property
    def vocab_size(self):
        """Vocabulary size."""
        return self._vocab_size
