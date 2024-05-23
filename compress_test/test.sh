#!/bin/bash

MODELS="cnn_not_optimised-default_opt.tflite cnn_not_optimised-int8_quant.tflite cnn_not_optimised-no_opt.tflite cnn_pruned-default_opt.tflite cnn_pruned-int8_quant.tflite cnn_pruned-no_opt.tflite"

for model in $MODELS
do
    zip ${model}.zip $model
done

ls -la | grep .zip