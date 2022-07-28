import numpy as np
import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

class BertForMultilabelSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
      super().__init__(config)

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict)

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels),
                            labels.float().view(-1, self.num_labels))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions)

def inference(sentences, num_labels):

    n_sentences = len(sentences)

    model = BertForMultilabelSequenceClassification.from_pretrained("bert-base-uncased",
                                                                    num_labels=num_labels,
                                                                    output_attentions=False,
                                                                    output_hidden_states=False)

    model.load_state_dict(torch.load('./finetuned_BERT_epoch_2.model', map_location=torch.device('cpu')))


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',
                                              do_lower_case=True)
    encoded_input = tokenizer.batch_encode_plus(
        sentences,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )
    output = model(**encoded_input)
    label = torch.argmax(output[0], dim=1)
    # print(np.asarray(sentences).T.reshape(2,1).shape)
    # print(output[0].detach().numpy().shape)
    return np.concatenate((np.asarray(sentences).T.reshape(n_sentences,1), label.numpy().reshape(n_sentences,1), output[0].detach().numpy()), axis=1)
