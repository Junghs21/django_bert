import re
import numpy as np
import tensorflow as tf


def predict_sentiment(sentence, tokenizer, model):

    SEQ_LEN = 128

    #Tokenizing / Tokens to sequence numbers / Padding
    encoded_dict = tokenizer.encode_plus(text=re.sub("[^\s0-9a-zA-Zㄱ-ㅎㅏ-ㅣ가-힣]", "", sentence),
                                         padding='max_length',
                                         truncation = True,
                                         max_length=SEQ_LEN) # SEQ_LEN == 128

    token_ids = np.array(encoded_dict['input_ids']).reshape(1, -1) # shape == (1, 128) : like appending to a list
    token_masks = np.array(encoded_dict['attention_mask']).reshape(1, -1)
    token_segments = np.array(encoded_dict['token_type_ids']).reshape(1, -1)

    new_inputs = (token_ids, token_masks, token_segments)

    #Prediction
    prediction = model.predict(new_inputs)
    predicted_probability = np.round(np.max(prediction) * 100, 2)
    predicted_class = ['부정', '긍정'][np.argmax(prediction, axis=1)[0]]
    result = "{}% 확률로 {} 리뷰입니다.".format(predicted_probability, predicted_class)

    return result
