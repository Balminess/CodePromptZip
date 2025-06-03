import torch
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput
from torch import nn
from typing import Optional, Union, Tuple
from transformers import T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput
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
        self.vocab_size = 32100 #vocab_size
    
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





class T5ForConditionalGenerationWithCopyMech(T5ForConditionalGeneration):
    
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.copy_module = CopyMechModule(config.d_model, config.vocab_size)
        
        self.post_init()
    
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
        output_attentions=True,
        output_hidden_states: Optional[bool] = None,
        return_dict=True,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
s
        Returns:
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=True,
                output_hidden_states=output_hidden_states,
                return_dict=True,
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
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        # print(decoder_outputs)
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
            p_gen, cp_logits = self.copy_module.forward(input_ids, cross_attentions, hidden_states, outputs)
            p_copy = 1 - p_gen
            logits = p_gen * lm_logits + p_copy * cp_logits
        else:
            # Inference (Step-by-step Decoding)
           
            cross_attentions = decoder_outputs.cross_attentions[-1].mean(dim=1)
            # print(input_ids.shape,cross_attentions.shape,hidden_states.shape,outputs.shape)
            p_gen, cp_logits = self.copy_module.forward(input_ids, cross_attentions, encoder_outputs[0], outputs)
            p_copy = 1 - p_gen
            logits = p_gen * lm_logits + p_copy * cp_logits

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            # labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
