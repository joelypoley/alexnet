echo "Submitting a Cloud ML Engine job..."

REGION="us-central1"
TIER="BASIC_GPU" # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1
BUCKET="joellaity" # change to your bucket name

CURRENT_DATE_TIME="`date +%Y_%m_%d_%H_%M_%S`"
MODEL_NAME=${CURRENT_DATE_TIME}

PACKAGE_PATH=trainer # this can be a gcs location to a zipped and uploaded package
TRAIN_FILES='gs://joellaity/imagenet2012_tfrecords/train-*-1024'
EVAL_FILES='gs://joellaity/imagenet2012_tfrecords/val-*-0128'
MODEL_DIR=gs://joellaity/model_dir/${MODEL_NAME}

CURRENT_DATE=`date +%Y%m%d_%H%M%S`
JOB_NAME=train_${MODEL_NAME}_${TIER}_${CURRENT_DATE}
#JOB_NAME=tune_${MODEL_NAME}_${CURRENT_DATE} # for hyper-parameter tuning jobs

gcloud ml-engine jobs submit training ${JOB_NAME} \
        --job-dir=${MODEL_DIR} \
        --region=${REGION} \
        --scale-tier=${TIER} \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH}  \
        --runtime-version=1.13 \
        -- \
        --train=${TRAIN_FILES} \
        --batch-size=128\
        --eval=${EVAL_FILES} \
        --learning-rate=0.01\
        --throttle-secs=1800\
        #--config=config.yaml \



echo "To view tensorboard type"
echo "tensorboard --logdir=$MODEL_DIR"
# notes:
# use --packages instead of --package-path if gcs location
# add --reuse-job-dir to resume training