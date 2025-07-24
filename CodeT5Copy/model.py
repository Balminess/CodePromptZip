import torch
from transformers import T5ForConditionalGeneration
from torch import nn
from dataclasses import dataclass
from typing import Optional, Union, Tuple, Dict, Any, List
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
        self.vocab_size = 32110 #vocab_size
    
    def forward(
        self,
        decoder_input_embeds: torch.FloatTensor,         # shape: (B, T, H)
        sequence_output: torch.FloatTensor,               # shape: (B, T, H)
        cross_attentions: torch.FloatTensor,             # shape: (B, T, S)
        input_ids_to_copy: torch.LongTensor,                 # shape: (B, S)
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:

        batch_size, seq_length = input_ids_to_copy.size(0), input_ids_to_copy.size(1)
        total_states = torch.cat((decoder_input_embeds, sequence_output), dim=-1)
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

    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
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

        # Decode
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
            return_dict=True,
            cache_position=cache_position,
        )

        sequence_output = decoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
 
        
    
            # Training
            # decoder_input_ids = decoder_input_ids[:, :-1] #this line cause [4,255,768]
        decoder_input_embeds = self.decoder.embed_tokens(decoder_input_ids)
        # decoder_input_embeds *= self.decoder.embed_scale # T5 noscale
        print(decoder_input_embeds.shape)
        cross_attentions = decoder_outputs.cross_attentions[-1].mean(dim=1)  # (B, T, S)
      
        p_gen, cp_logits = self.copy_module.forward(
            decoder_input_embeds,            # decoder input embeds
            sequence_output,                         # decoder output
            cross_attentions,               # attention
            input_ids                        # encoder input ids
        )
        p_copy = 1 - p_gen
        logits = p_gen * lm_logits + p_copy * cp_logits


        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
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
