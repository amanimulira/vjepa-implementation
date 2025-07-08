# vjepa-implementation

Implementation of the VJEPA2 model, desinged to learn contextualized representations of video data and can be fine-tuned for specific tasks such as video classification.

## Architecture

CONTEXT ENCODER, transformer-based encoder that takes raw video pixel values w/ masks as input and generates contextualized representations.

PREDICTOR, module that takes the output of the CONTEXT ENCODER and predicts the full feature representaiton for the masked tokens.

TARGET ENCODER, again transformer-based encoder that take raw pixel values w/o masks as input and generates contextualized represenations.

