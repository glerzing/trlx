import cProfile
import copy
import gc
import numpy as np
import sys
import tempfile
import torch
import transformers
import unittest

from peft.utils.config import PeftType, TaskType
from time import time
from trlx.data.configs import TokenizerConfig
from trlx.data.default_configs import default_ppo_config, default_ilql_config, default_sft_config, ModelConfig
from trlx.models.modeling_ppo import (
    AutoModelForCausalLMWithHydraValueHead,
    AutoModelForSeq2SeqLMWithHydraValueHead,
)
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from typing import Optional

PPO = "ppo"
ILQL = "ilql"
SFT = "sft"
TRAINING_TYPES = [PPO, ILQL, SFT]

CAUSAL = "causal"
SEQ2SEQ = "seq2seq"
TASK_TYPES = [CAUSAL, SEQ2SEQ]

MODEL_TASK_TYPE = {
    "gpt2": CAUSAL,
    # "EleutherAI/pythia-410m-deduped": CAUSAL,
    # "EleutherAI/pythia-160m": CAUSAL,
    # "facebook/opt-125m": CAUSAL,
    "t5-small": SEQ2SEQ,
}
MODELS_TO_TEST = list(MODEL_TASK_TYPE.keys())

PEFT_CONFIGS_TO_TEST = [PeftType.LORA, PeftType.PROMPT_TUNING, PeftType.PREFIX_TUNING]

ALL_TEST_COMBINATIONS = [
    [training_type, model_path, peft_type]
    for training_type in TRAINING_TYPES
    for model_path in MODELS_TO_TEST
    for peft_type in PEFT_CONFIGS_TO_TEST
]

class TestPpoPeftLora(unittest.TestCase):

    def setUp(self):
        magic_number = 42
        np.random.seed(magic_number)
        torch.manual_seed(magic_number)
        torch.cuda.manual_seed_all(magic_number)

    def tearDown(self):
        gc.collect()  # Try to free up memory

    def _create_model(
            self,
            training_type: str,
            model_path: str,
            tokenizer_path: str,
            task_type: str,
            peft_type: Optional[str]
    ):
        start = time()
        peft_config = self._get_peft_config(peft_type, task_type, tokenizer_path) if peft_type else None
        config = self._get_config(training_type, model_path, tokenizer_path, task_type, peft_config)
        print("config time:", time() - start)

        start = time()
        self.trainer = AcceleratePPOTrainer(config)
        print("trainer time:", time() - start)
        self.model = self.trainer.model

        start = time()
        self._create_inputs(tokenizer_path, task_type)
        print("inputs time:", time() - start)

    def _create_inputs(self, tokenizer_path, task_type):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

        if task_type == CAUSAL:
            self.inputs = self.tokenizer(
                "Once upon a time there was a happy goose named Louis. He liked to eat bananas.",
                return_tensors="pt",
            )
        elif task_type == SEQ2SEQ:
            self.encoder_text = "Translate this text to French: Hello, my dog is cute"
            self.decoder_text = "Bonjour, mon chien est mignon"
            encoder_inputs = self.tokenizer(
                self.encoder_text, return_tensors="pt"
            )
            decoder_inputs = self.tokenizer(self.decoder_text, return_tensors="pt")
            self.inputs = {
                **encoder_inputs,
                "decoder_input_ids": decoder_inputs.input_ids,
                "decoder_attention_mask": decoder_inputs.attention_mask,
                "decoder_inputs_embeds": self.model.base_model.word_embeddings(decoder_inputs.input_ids) # TODO: remove when the issue https://github.com/huggingface/peft/issues/439 is fixed
                                         if hasattr(self.model.base_model, "word_embeddings") else None
            }
        else:
            # Classification tasks not implemented
            raise NotImplementedError

    def _get_config(
            self,
            training_type,
            model_path: str,
            tokenizer_path: str,
            task_type: str,
            peft_config: Optional[str]
    ):
        if training_type == PPO:
            config = default_ppo_config()
        elif training_type == ILQL:
            config = default_ilql_config()
        elif training_type == SFT:
            config = default_sft_config()
        else:
            raise ValueError(f"Training type {training_type} not recognized.")

        config.tokenizer = TokenizerConfig(tokenizer_path=tokenizer_path)
        config.model = ModelConfig(
            model_path=model_path,
            num_layers_unfrozen=1,
            peft_config=peft_config,
            model_arch_type=task_type
        )
        config.train.tracker = None

        return config

    def _get_peft_config(self, peft_type: str, task_type: str, tokenizer_name_or_path: str = None):
        assert task_type in TASK_TYPES
        task_type = TaskType.CAUSAL_LM if task_type == "causal" else TaskType.SEQ_2_SEQ_LM

        if peft_type == PeftType.LORA:
            return {
                "peft_type": peft_type,
                "task_type": task_type,
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
            }
        elif peft_type == PeftType.PREFIX_TUNING:
            return {
                "peft_type": peft_type,
                "task_type": task_type,
                "num_virtual_tokens": 10,
            }
        elif peft_type == PeftType.PROMPT_TUNING:
            return {
                "peft_type": peft_type,
                "task_type": task_type,
                "prompt_tuning_init": "RANDOM",
                "num_virtual_tokens": 10,
                "tokenizer_name_or_path": tokenizer_name_or_path,
            }
        else:
            raise NotImplementedError

    def _backprop(self, model):
        output = model(**self.inputs, return_dict=True).logits
        # Just apply an arbitrary loss to cause whatever change in the model's parameters.
        # This loss doesn't make sense, but it causes a gradient, so it's fine.
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output[0][-1][:1],
            torch.tensor([0.53]),
        )
        loss.backward()

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.step()

        return model

    def test_backpropagation_and_disabling(self):
        for task_type in TASK_TYPES:
            for model_path in LM_PATHS[task_type]:
                for peft_type in PEFT_CONFIGS_TO_TEST:
                    self._create_model(model_path, model_path, task_type, peft_type)
                    old_logits = self.model(**self.inputs, return_dict=True).logits
                    initial_model_state_dict = copy.deepcopy(self.model.state_dict())

                    self._backprop(self.model)
                    self._backprop(self.model)
                    new_logits = self.model(**self.inputs, return_dict=True).logits
                    new_model_state_dict = self.model.state_dict()

                    # Check that the backpropagation affected the predictions
                    self.assertFalse(torch.equal(old_logits, new_logits))

                    # Check that only the peft adapter layers are modified by the backpropagation
                    self.assertEqual(initial_model_state_dict.keys(), new_model_state_dict.keys())
                    for name in initial_model_state_dict.keys():
                        parameters_equal = torch.equal(initial_model_state_dict[name], new_model_state_dict[name])
                        if "lora" in name or "prompt" in name:
                            self.assertFalse(parameters_equal)
                        else:
                            self.assertTrue(parameters_equal)

                    # Check Lora enabling and disabling
                    if "LORA" in peft_type:
                        # If disabling the Lora adapter restores the original logits,
                        # this shows that the backpropagation only affected the Lora adapter
                        self.lora_model = self.model.base_model.base_model
                        self.lora_model.disable_adapter_layers()
                        new_logits = self.model(**self.inputs, return_dict=True).logits
                        self.assertTrue(torch.equal(old_logits, new_logits))

                        # Re-enabling the Lora adapter should make the 2 models different again
                        self.lora_model.enable_adapter_layers()
                        new_logits = self.model(**self.inputs, return_dict=True).logits
                        self.assertFalse(torch.equal(old_logits, new_logits))

    def test_forward_hydra(self):
        """Test that hydra heads work and give similar logits to the model without any fine-tuning."""
        for task_type in TASK_TYPES:
            for model_path in LM_PATHS[task_type]:
                for peft_type in PEFT_CONFIGS_TO_TEST:
                    self._create_model(model_path, model_path, task_type, peft_type)

                    # TODO: For T5, you need to specify either decoder_input_ids or decoder_inputs_embeds but not both.
                    # Clean this patch after https://github.com/huggingface/peft/issues/439 is solved by using only self.inputs
                    patched_inputs = self.inputs.copy()
                    if task_type == SEQ2SEQ and peft_type != PeftType.LORA:
                        patched_inputs.pop("decoder_inputs_embeds", None)

                    logits_without_peft = self.model.base_model.base_model(**patched_inputs, return_dict=True).logits
                    logits_before_backpropagation = self.model(**self.inputs, return_dict=True).logits

                    self._backprop(self.model)

                    # forward_hydra should return the same logits as the original model
                    new_logits_from_hydra = self.model.forward_hydra(**patched_inputs, return_dict=True).logits
                    self.assertTrue(torch.equal(logits_without_peft, new_logits_from_hydra))

                    if "LORA" in peft_type:
                        # True because the Lora adapter initially does not modify the output
                        self.assertTrue(torch.equal(logits_before_backpropagation, new_logits_from_hydra))
                    else:
                        # False because the initial prompt before backpropagation
                        # was used to calculate logits_before_backpropagation, but not for new_logits_from_hydra.
                        self.assertFalse(torch.equal(logits_before_backpropagation, new_logits_from_hydra))

    def test_generate(self):
        for training_type, model_path, peft_type in ALL_TEST_COMBINATIONS:
            task_type = MODEL_TASK_TYPE[model_path]
            if task_type == SEQ2SEQ and peft_type == PeftType.PROMPT_TUNING:
                return  # TODO: remove when seq2seq PROMPT_TUNING generate is implemented on on the peft repository
            prof = cProfile.Profile()
            prof.enable()
            start = time()
            self._create_model(training_type, model_path, model_path, task_type, peft_type)
            print("time1: ", time() - start)
            self._backprop(self.model)
            print("time2: ", time() - start)

            # Check that generate works, and that it's deterministic
            output1 = self.model.generate(**self.inputs, temperature=0.0,
                                          pad_token_id=self.tokenizer.eos_token_id)
            self.model.eval()
            output2 = self.model.generate(**self.inputs, temperature=0.0,
                                          pad_token_id=self.tokenizer.eos_token_id)
            self.assertTrue(torch.equal(output1, output2))
            print("time3: ", time() - start)
            prof.disable()
            import pstats
            stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
            stats.print_stats(10)  # top 10 rows
            raise


    def test_from_pretrained(self):
        pass

    def test_lora_modules_to_save(self):
        """
        Test the special Lora config option 'modules_to_save'.
        It allows not to freeze some non-lora modules, and its implementation is a bit tricky.
        """
        pass

    def test_ilql(self):
        pass

    def test_sft(self):
        pass

    def test_save_load(self):
        for training_type, model_path, peft_type in ALL_TEST_COMBINATIONS:
            task_type = MODEL_TASK_TYPE[model_path]
            self._create_model(training_type, model_path, model_path, task_type, peft_type)
            self._backprop(self.model)

            with tempfile.TemporaryDirectory() as tmp_dir_name:
                self.model.save_pretrained(tmp_dir_name)
                if task_type == TaskType.CAUSAL_LM:
                    model_type = AutoModelForCausalLMWithHydraValueHead
                else:
                    model_type = AutoModelForSeq2SeqLMWithHydraValueHead

                loaded_model = model_type.from_pretrained(
                    model_path,
                    peft_adapter_path=tmp_dir_name,
                )

            # Check that the loaded model state dict is the same as the saved model state dict
            model_state_dict = self.model.state_dict()
            loaded_state_dict = loaded_model.state_dict()
            for name in loaded_state_dict.keys():
                self.assertTrue(torch.equal(model_state_dict[name], loaded_state_dict[name]))

            self.assertTrue(
                torch.equal(self.model(**self.inputs, return_dict=True).logits,
                            loaded_model(**self.inputs, return_dict=True).logits)
            )

            self.assertTrue(
                torch.equal(self.model(**self.inputs, return_dict=True).value,
                            loaded_model(**self.inputs, return_dict=True).value)
            )

            self.assertTrue(
                torch.equal(self.model.forward_hydra(**self.inputs, return_dict=True).logits,
                            loaded_model.forward_hydra(**self.inputs, return_dict=True).logits)
            )

    def test_save_load_without_peft(self):
        """Similar to test_save_load, but with peft not installed. Should not raise any error."""
        with unittest.mock.patch.dict(sys.modules, {"peft": None}):
            for training_type, model_path, peft_type in ALL_TEST_COMBINATIONS:
                task_type = MODEL_TASK_TYPE[model_path]
                self._create_model(training_type, model_path, model_path, task_type, peft_type=None)
                self._backprop(self.model)

                with tempfile.TemporaryDirectory() as tmp_dir_name:
                    self.model.save_pretrained(tmp_dir_name)
                    if task_type == TaskType.CAUSAL_LM:
                        model_type = AutoModelForCausalLMWithHydraValueHead
                    else:
                        model_type = AutoModelForSeq2SeqLMWithHydraValueHead

                    loaded_model = model_type.from_pretrained(tmp_dir_name)

                # Check that the loaded model state dict is the same as the saved model state dict
                model_state_dict = self.model.state_dict()
                loaded_state_dict = loaded_model.state_dict()
                for name in loaded_state_dict.keys():
                    self.assertTrue(torch.equal(model_state_dict[name], loaded_state_dict[name]))

                self.assertTrue(
                    torch.equal(self.model(**self.inputs, return_dict=True).logits,
                                loaded_model(**self.inputs, return_dict=True).logits)
                )

                self.assertTrue(
                    torch.equal(self.model(**self.inputs, return_dict=True).value,
                                loaded_model(**self.inputs, return_dict=True).value)
                )

                self.assertTrue(
                    torch.equal(self.model.forward_hydra(**self.inputs, return_dict=True).logits,
                                loaded_model.forward_hydra(**self.inputs, return_dict=True).logits)
                )

    def test_peft_not_installed_error(self):
        """Having peft installed should be needed in those cases, otherwise cause a ModuleNotFoundError"""
        with unittest.mock.patch.dict(sys.modules, {"peft": None}):
            with self.assertRaises(ModuleNotFoundError):
                config = default_ppo_config()
                config.model = ModelConfig(
                    model_path="gpt2",
                    num_layers_unfrozen=1,
                    peft_config={"peft_type": "LORA"},
                    model_arch_type="causal"
                )

            with self.assertRaises(ModuleNotFoundError):
                AutoModelForCausalLMWithHydraValueHead.from_pretrained("gpt2", peft_adapter_path="gpt2")

    def test_8bits(self):
        pass


test = TestPpoPeftLora()
test.setUp()
test.test_save_load()
AutoModelForCausalLMWithHydraValueHead.from_pretrained("gpt2")
# test = TestPpoPeftLora()
# test.setUp()
# test.test_generate()
# raise