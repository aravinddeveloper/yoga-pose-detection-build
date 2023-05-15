import os
import boto3
import sagemaker
import sagemaker.session
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.workflow.functions import Join

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
BUCKET_NAME = "aravind-aws-ml-sagemaker"
PROCESSING_PATH = "/opt/ml/processing"
MODEL_PATH = "/opt/ml/model"


def get_session(region, default_bucket=BUCKET_NAME):
  
    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline( region,
                 default_bucket=BUCKET_NAME,
                 role=None,
                 model_package_group_name="YogaPoseDetectionPackageGroup",  # Choose any name
                 pipeline_name="yogapose-detection-pipeline",  # You can find your pipeline name in the Studio UI (project -> Pipelines -> name)
                 base_job_prefix="yogapose"):
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)
        
    # Parameters for preprocessing pipeline execution
    print("Setting parameters for pipeline")
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.c4.2xlarge"
    )
    epochs = ParameterInteger(name="Epochs", default_value=1)
    batch_size = ParameterInteger(name="BatchSize", default_value=5)
    train_folder = ParameterString(
        name="TrainFolderName",
        default_value="train"  # Change this to point to the s3 location of your raw input data.
    )
    val_folder = ParameterString(
        name="ValidationFolderName",
        default_value="val"  # Change this to point to the s3 location of your raw input data.
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",  # ModelApprovalStatus can be set to a default of "Approved" if you don't want manual approval.
    )
    input_data_path = ParameterString(
        name="InputDataUrl",
        default_value=Join(on="",values=["s3://",BUCKET_NAME,"/input"]) #f"s3://{BUCKET_NAME}/input",  # Change this to point to the s3 location of your raw input data.
    )
    processed_data_path = ParameterString(
        name="OutputDataUrl",
        default_value=Join(on="",values=["s3://",BUCKET_NAME,"/processed"]) #f"s3://{BUCKET_NAME}/processed",  # Change this to point to the s3 location of your raw input data.
    )
    model_path = ParameterString(
        name="ModelPath",
        default_value=Join(on="",values=["s3://",BUCKET_NAME,"/model"]) #f"s3://{BUCKET_NAME}/model",  # Change this to point to the s3 location of your raw input data.
    )
    
    cache_config = CacheConfig(
        enable_caching=True,
        expire_after="T1H"
    )
    
    print("Setting pytorch processor")
    processor = PyTorchProcessor(role=role,
                                 framework_version="1.8",
                                 instance_type=processing_instance_type,
                                 instance_count=processing_instance_count,
                                 sagemaker_session=sagemaker_session)

    processing_step = ProcessingStep(name="pre-processing-step",
                                     processor=processor,
                                     inputs=ProcessingInput(input_name="raw-data",
                                                            source=input_data_path,
                                                            destination=Join(on="",values=[PROCESSING_PATH,"/input"])),
                                     code=os.path.join(BASE_DIR,"processing.py"),
                                     outputs=[ProcessingOutput(output_name=train_folder,
                                                               source=Join(on="",values=[PROCESSING_PATH,"/",train_folder]),
                                                               destination=Join(on="",values=[processed_data_path,"/",train_folder])),
                                              ProcessingOutput(output_name=val_folder,
                                                               source=Join(on="",values=[PROCESSING_PATH,"/",val_folder]),
                                                               destination=Join(on="",values=[processed_data_path,"/",val_folder]))],
                                    cache_config=cache_config)
    #Env variables
    env_variables = {
        "SM_NUM_GPUS":"1",
        "SM_MODEL_DIR":f"{MODEL_PATH}/model/",
        "SM_OUTPUT_DATA_DIR":f"{MODEL_PATH}/output/",
        "SM_CHANNEL_TRAIN":f"{MODEL_PATH}/train/",
        "SM_CHANNEL_VAL":f"{MODEL_PATH}/val/"
    }
    
    print("Setting Training estimator")
    pt_estimator = PyTorch(
        entry_point="train.py",
        role=get_execution_role(),
        source_dir=BASE_DIR,
        framework_version="1.8.0",
        py_version="py3",
        instance_count=training_instance_count,
        instance_type=training_instance_type,
        use_spot_instances=True,
        checkpoint_s3_uri=Join(on="",values=["s3://",BUCKET_NAME,"/",base_job_prefix,"/checkpoints/"]),
        checkpoint_local_path="/opt/ml/checkpoints/",
        max_run=3600,
        max_wait=3800,
        environment=env_variables,
        sagemaker_session=pipeline_session)
    
    pt_estimator.set_hyperparameters(epochs=epochs,
                                     batch_size=batch_size)
    
    step_train = TrainingStep(
        name="training-step",
        estimator=pt_estimator,
        inputs={
            train_folder: TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[train_folder].S3Output.S3Uri
            ),
            val_folder: TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[val_folder].S3Output.S3Uri
            )
        },
        depends_on=processing_step,
        cache_config=cache_config)
    
    # Processing step for evaluation
    print("Setting model evaluation")
    model_eval = PyTorchProcessor(role=role,
                                 framework_version="1.8",
                                 instance_type=processing_instance_type,
                                 instance_count=processing_instance_count,
                                 sagemaker_session=sagemaker_session)
    
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    
    step_eval = ProcessingStep(
        name="evaluation-step",
        processor=model_eval,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["eval"].S3Output.S3Uri,
                destination="/opt/ml/processing/eval",
            )
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/eval"),
        ],
        code=os.path.join(BASE_DIR, "evaluation.py"),
        property_files=[evaluation_report]
    )
    
    # Register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )
    
    # Register model step that will be conditionally executed
    print("Registering model")
    step_register = RegisterModel(
        name="register-model",
        estimator=pt_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["image/png","image/jpg","image/jpeg"],
        response_types=["application/json"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics
    )
    
    accuracy_score = JsonGet(
        step_name=step_eval.name,
        property_file=evaluation_report,
        json_path="multiclass_classification_metrics.accuracy.value"
    )
    
    condition = ConditionGreaterThanOrEqualTo(left=accuracy_score,right=0.8)
    step_condition = ConditionStep(
        name="evaluation-condition-check",
        conditions=[condition],
        if_steps=[step_register],
        else_steps=[]
    )
    
    # Pipeline instance
    print("Building final pipeline")
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_count,
            processing_instance_type,
            training_instance_count,
            training_instance_type,
            epochs,
            batch_size,
            train_folder,
            val_folder,
            model_approval_status,
            input_data_path,
            processed_data_path,
            model_path
        ],
        steps=[processing_step, step_train, step_eval, step_condition],
        sagemaker_session=sagemaker_session,
    )
    return pipeline
    
    