import copy
import gc
import numpy as np
import torch
import transformers
import tempfile
import unittest

from peft import PeftModel
from peft.utils.config import PeftType, TaskType
from trlx.data.configs import TokenizerConfig
from trlx.data.default_configs import default_ppo_config, ModelConfig
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

LM_PATHS = {
    TaskType.CAUSAL_LM: ["gpt2", "EleutherAI/pythia-160m", "facebook/opt-125m"],
    TaskType.SEQ_2_SEQ_LM: ["t5-small"],
}
PEFT_CONFIGS_TO_TEST = [PeftType.LORA, PeftType.PROMPT_TUNING, PeftType.PREFIX_TUNING]

class TestPpoPeftLora(unittest.TestCase):

    def setUp(self):
        magic_number = 42
        np.random.seed(magic_number)
        torch.manual_seed(magic_number)
        torch.cuda.manual_seed_all(magic_number)

    def tearDown(self):
        gc.collect()  # Try to free up memory

    def _get_peft_kwargs(self, peft_type: str, task_type: str, tokenizer_name_or_path: str = None):
        assert task_type in LM_PATHS.keys()

        if peft_type == "LORA":
            return {
                "peft_type": "LORA",
                "task_type": task_type,
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.0,
            }
        elif peft_type == "PREFIX_TUNING":
            return {
                "peft_type": peft_type,
                "task_type": task_type,
                "num_virtual_tokens": 10,
            }
        elif peft_type == "PROMPT_TUNING":
            return {
                "peft_type": peft_type,
                "task_type": task_type,
                "prompt_tuning_init": "RANDOM",
                "num_virtual_tokens": 10,
                "tokenizer_name_or_path": tokenizer_name_or_path,
            }
        else:
            raise NotImplementedError

    def _create_model(self, model_path: str, tokenizer_path: str, peft_type: str, task_type: str):
        peft_kwargs = self._get_peft_kwargs(peft_type, task_type, tokenizer_path)
        config = default_ppo_config()
        config.tokenizer = TokenizerConfig(tokenizer_path=tokenizer_path)
        config.model = ModelConfig(
            model_path=model_path,
            num_layers_unfrozen=1,
            peft_kwargs=peft_kwargs,
            model_arch_type="causal" if task_type == TaskType.CAUSAL_LM else "seq2seq"
        )
        config.train.tracker = None

        self.trainer = AcceleratePPOTrainer(config)
        self.model = self.trainer.model

        self._create_inputs(tokenizer_path, task_type)

    def _create_inputs(self, tokenizer_path, task_type):
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

        if task_type == TaskType.CAUSAL_LM:
            self.inputs = self.tokenizer(
                "Once upon a time there was a happy goose named Louis. He liked to eat bananas.",
                return_tensors="pt",
            )
        elif task_type == TaskType.SEQ_2_SEQ_LM:
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
        for task_type in [TaskType.CAUSAL_LM, TaskType.SEQ_2_SEQ_LM]:
            for model_path in LM_PATHS[task_type]:
                for peft_type in PEFT_CONFIGS_TO_TEST:
                    self._create_model(model_path, model_path, peft_type, task_type)
                    old_logits = self.model(**self.inputs, return_dict=True).logits
                    initial_model_state_dict = copy.deepcopy(self.model.state_dict())

                    self._backprop(self.model)
                    self._backprop(self.model)
                    new_logits = self.model(**self.inputs, return_dict=True).logits

                    new_model_state_dict = self.model.state_dict()
                    self.assertEqual(initial_model_state_dict.keys(), new_model_state_dict.keys())
                    for name in initial_model_state_dict.keys():
                        # Only the peft adapter layers must have been modified by the backpropagation
                        parameters_equal = torch.equal(initial_model_state_dict[name], new_model_state_dict[name])
                        if "lora" in name or "prompt" in name:
                            self.assertFalse(parameters_equal)
                        else:
                            self.assertTrue(parameters_equal)

                    # Check that the backpropagation affected the predictions
                    self.assertFalse(torch.equal(old_logits, new_logits))

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
        for task_type in [TaskType.CAUSAL_LM, TaskType.SEQ_2_SEQ_LM]:
            for model_path in LM_PATHS[task_type]:
                for peft_type in PEFT_CONFIGS_TO_TEST:
                    self._create_model(model_path, model_path, peft_type, task_type)

                    # TODO: For T5, you need to specify either decoder_input_ids or decoder_inputs_embeds but not both.
                    # Clean this patch after https://github.com/huggingface/peft/issues/439 is solved by using only self.inputs
                    patched_inputs = self.inputs.copy()
                    if task_type == TaskType.SEQ_2_SEQ_LM and peft_type != PeftType.LORA:
                        patched_inputs.pop("decoder_inputs_embeds", None)

                    logits_without_peft = self.model.base_model.base_model(**patched_inputs, return_dict=True).logits
                    logits_before_backpropagation = self.model(**self.inputs, return_dict=True).logits

                    self._backprop(self.model)

                    # forward_hydra should return the same logits as the original model
                    new_logits_from_hydra = self.model.forward_hydra(**patched_inputs, return_dict=True).logits
                    if "LORA" in peft_type:
                        # True because the Lora adapter initially does not modify the output
                        self.assertTrue(torch.equal(logits_before_backpropagation, new_logits_from_hydra))
                    else:
                        # False because the initial prompt before backpropagation
                        # was used to calculate logits_before_backpropagation, but not for new_logits_from_hydra.
                        self.assertFalse(torch.equal(logits_before_backpropagation, new_logits_from_hydra))

                    self.assertTrue(torch.equal(logits_without_peft, new_logits_from_hydra))

    def test_save_load(self):
        for task_type in [TaskType.CAUSAL_LM, TaskType.SEQ_2_SEQ_LM]:
            for model_path in LM_PATHS[task_type]:
                for peft_type in PEFT_CONFIGS_TO_TEST:
                    self._create_model(model_path, model_path, peft_type, task_type)
                    self._backprop(self.model)

                    with tempfile.TemporaryDirectory() as tmp_dir_name:
                        self.model.save_pretrained(tmp_dir_name)
                        if task_type == TaskType.CAUSAL_LM:
                            loaded_model = AutoModelForCausalLM.from_pretrained(model_path)
                        else:
                            loaded_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
                        loaded_model = PeftModel.from_pretrained(loaded_model, tmp_dir_name)

                    # Check that the loaded model state dict is the same as the saved model state dict
                    model_state_dict = self.model.state_dict()
                    loaded_state_dict = loaded_model.state_dict()
                    for name in loaded_state_dict.keys():
                        if "v_head" not in name:
                            self.assertTrue(torch.equal(model_state_dict[name], loaded_state_dict[name]))

                    self.assertTrue(
                        torch.equal(self.model(**self.inputs, return_dict=True).logits,
                                    loaded_model(**self.inputs, return_dict=True).logits)
                    )

    def test_generate(self):
        for task_type in [TaskType.CAUSAL_LM, TaskType.SEQ_2_SEQ_LM]:
            for model_path in LM_PATHS[task_type]:
                for peft_type in PEFT_CONFIGS_TO_TEST:
                    if task_type == TaskType.SEQ_2_SEQ_LM and peft_type == PeftType.PROMPT_TUNING:
                        continue  # TODO: remove when seq2seq PROMPT_TUNING generate is implemented on on the peft repository

                    self._create_model(model_path, model_path, peft_type, task_type)
                    self._backprop(self.model)

                    # Check that generate works, and that it's deterministic
                    output1 = self.model.generate(**self.inputs, temperature=0.0,
                                                  pad_token_id=self.tokenizer.eos_token_id)
                    self.model.eval()
                    output2 = self.model.generate(**self.inputs, temperature=0.0,
                                                  pad_token_id=self.tokenizer.eos_token_id)
                    self.assertTrue(torch.equal(output1, output2))

# test = TestPpoPeftLora()
# test.setUp()
# test.test_save_load()
# raise