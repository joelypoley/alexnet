MODEL_DIR=trained_models
JS_MODEL_DIR=js_trained_models


MODEL_LOCATION=${MODEL_DIR}/$(ls ${MODEL_DIR} | tail -1)
EXPORTED_MODEL_LOCATION=${MODEL_LOCATION}/export/estimator/$(ls ${MODEL_LOCATION}/export/estimator | tail -1)
JS_MODEL_LOCATION=${JS_MODEL_DIR}/$(ls ${MODEL_DIR} | tail -1)


echo "Will convert the following graph"
saved_model_cli show \
        --dir=${EXPORTED_MODEL_LOCATION} \
        --all

echo "Converting: "
echo ${MODEL_LOCATION}
echo "And placing the js model in:"
echo ${JS_MODEL_LOCATION}

tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names="dense/BiasAdd" \
    ${EXPORTED_MODEL_LOCATION} \
    ${JS_MODEL_LOCATION}