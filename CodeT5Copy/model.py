import torch
from transformers import T5ForConditionalGeneration
from torch import nn
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict, Any, List
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, shift_tokens_right
from transformers.models.bart.configuration_bart import BartConfig
from transformers import T5Config
from transformers import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions, Seq2SeqLMOutput, ModelOutput
from transformers.utils import logging
from transformers import T5ForConditionalGeneration


logger = logging.get_logger(__name__)


class CopyMechModule(nn.Module):
    
    def __init__(self, transformer_hidden_size, vocab_size):
        super().__init__()
        self.p_gen_head = nn.Sequential(
            nn.Linear(transformer_hidden_size * 2, 1),
            nn.Sigmoid(),
        )
        self.vocab_size = 32107 #vocab_size
    
    def forward(
        self,
        input_ids_to_copy: torch.LongTensor,
        cross_attentions: torch.FloatTensor,
        src_hidden_states: torch.FloatTensor,
        tgt_hidden_states: torch.FloatTensor,
    ) -> torch.FloatTensor:
        batch_size, seq_length = input_ids_to_copy.size(0), input_ids_to_copy.size(1)
        context_vectors = cross_attentions @ src_hidden_states
        total_states = torch.cat((context_vectors, tgt_hidden_states), dim=-1)
        p_gen = self.p_gen_head(total_states)
        input_one_hot = input_ids_to_copy.new_zeros(batch_size, seq_length, self.vocab_size)
        input_one_hot.scatter_(-1, input_ids_to_copy[:, :, None], 1)
        input_one_hot = input_one_hot.float()
        logits = cross_attentions @ input_one_hot
        return p_gen, logits


@dataclass
class Seq2SeqLMOutputWithSrcIds(Seq2SeqLMOutput):
    src_input_ids: Optional[Tuple[torch.LongTensor]] = None


class T5ForConditionalGenerationWithCopyMech(T5ForConditionalGeneration):
    
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.copy_module = CopyMechModule(config.d_model, config.vocab_size)
        self.post_init()
    
    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)
        model_kwargs["src_input_ids"] = inputs_tensor
        
        return model_kwargs
    
    def _expand_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        attention_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[ModelOutput] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if "src_input_ids" in model_kwargs:
            src_input_ids = model_kwargs["src_input_ids"]
            model_kwargs["src_input_ids"] = src_input_ids.index_select(0, expanded_return_idx)
        
        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx.to(encoder_outputs.last_hidden_state.device)
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        return input_ids, model_kwargs
    
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "src_input_ids": kwargs["src_input_ids"],
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        src_input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutputWithSrcIds]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
                
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        hidden_states = encoder_outputs[0]
                
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)


        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs= decoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)   

        lm_logits = self.lm_head(outputs)
        # lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        
        if labels is not None:
            # Training
            cross_attentions = decoder_outputs.cross_attentions[-1].mean(dim=1)
            print(input_ids.shape,cross_attentions.shape,encoder_outputs.last_hidden_state.shape,outputs.shape)
            p_gen, cp_logits = self.copy_module.forward(input_ids, cross_attentions, encoder_outputs.last_hidden_state, outputs)
            p_copy = 1 - p_gen
            logits = p_gen * lm_logits + p_copy * cp_logits
        else:
            # Inference (Step-by-step Decoding)
            cross_attentions = decoder_outputs.cross_attentions[-1].mean(dim=1)
            p_gen, cp_logits = self.copy_module.forward(src_input_ids, cross_attentions, encoder_outputs[0], outputs)
            p_copy = 1 - p_gen
            logits = p_gen * lm_logits + p_copy * cp_logits

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
