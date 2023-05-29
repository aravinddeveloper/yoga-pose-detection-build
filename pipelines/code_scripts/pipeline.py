import os
import boto3
import sagemaker
import sagemaker.session
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.pytorch.estimator import PyTorch
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.workflow.functions import Join
from sagemaker.workflow.pipeline_context import PipelineSession

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
BUCKET_NAME = "aravind-aws-ml-sagemaker"
PROCESSING_PATH = "/opt/ml/processing"
MODEL_PATH = "/opt/ml/model"
train_folder_name="train"
val_folder_name="val"
eval_folder_name="eval"

def get_session(region, default_bucket=BUCKET_NAME,pipe_session=False):
  
    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    if pipe_session:
        return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
        )
    else:
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
    eval_folder = ParameterString(
        name="EvaluationFolderName",
        default_value="eval"  # Change this to point to the s3 location of your raw input data.
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",  # ModelApprovalStatus can be set to a default of "Approved" if you don't want manual approval.
    )
    input_data_path = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://{BUCKET_NAME}/input" #Join(on="",values=["s3://",BUCKET_NAME,"/input"]) #f"s3://{BUCKET_NAME}/input",  # Change this to point to the s3 location of your raw input data.
    )
    processed_data_path = ParameterString(
        name="OutputDataUrl",
        default_value=f"s3://{BUCKET_NAME}/processed" #Join(on="",values=["s3://",BUCKET_NAME,"/processed"]) ,  # Change this to point to the s3 location of your raw input data.
    )
    model_path = ParameterString(
        name="ModelPath",
        default_value=f"s3://{BUCKET_NAME}/model" #Join(on="",values=["s3://",BUCKET_NAME,"/model"]) #,  # Change this to point to the s3 location of your raw input data.
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
    
    print("setting processing step args")
    proc_args = processor.run(inputs=[ProcessingInput(input_name="raw-data",
                                                      source=input_data_path,
                                                      destination=f"{PROCESSING_PATH}/input")],
                              code=os.path.join(BASE_DIR,"processing.py"),
                              outputs=[ProcessingOutput(output_name=train_folder_name,
                                                        source=f"{PROCESSING_PATH}/{train_folder_name}",
                                                        destination=Join(on="",values=[processed_data_path,"/",train_folder_name])),
                                       ProcessingOutput(output_name=val_folder_name,
                                                        source=f"{PROCESSING_PATH}/{val_folder_name}",
                                                        destination=Join(on="",values=[processed_data_path,"/",val_folder_name])),
                                       ProcessingOutput(output_name=eval_folder_name,
                                                        source=f"{PROCESSING_PATH}/{eval_folder_name}",
                                                        destination=Join(on="",values=[processed_data_path,"/",eval_folder_name]))],
                              arguments=["--train-test-split_ratio", 0.1, 
                                         "--process-input-folder","input",
                                         "--process-train-folder",train_folder_name,
                                         "--process-val-folder",val_folder_name,
                                         "--process-eval-folder",eval_folder_name])
    # proc_args = processor.run(inputs=[ProcessingInput(input_name="raw-data",
    #                                                   source=input_data_path,
    #                                                   destination=Join(on="",values=[PROCESSING_PATH,"/input"]))],
    #                           code=os.path.join(BASE_DIR,"processing.py"),
    #                           outputs=[ProcessingOutput(output_name=train_folder_name,
    #                                                     source=Join(on="",values=[PROCESSING_PATH,"/",train_folder_name]),
    #                                                     destination=Join(on="",values=[processed_data_path,"/",train_folder_name])),
    #                                    ProcessingOutput(output_name=val_folder_name,
    #                                                     source=Join(on="",values=[PROCESSING_PATH,"/",val_folder_name]),
    #                                                     destination=Join(on="",values=[processed_data_path,"/",val_folder_name])),
    #                                    ProcessingOutput(output_name=eval_folder_name,
    #                                                     source=Join(on="",values=[PROCESSING_PATH,"/",eval_folder_name]),
    #                                                     destination=Join(on="",values=[processed_data_path,"/",eval_folder_name]))],
    #                           arguments=["--train-test-split_ratio", 0.1, 
    #                                      "--process-input-folder","input",
    #                                      "--process-train-folder","train",
    #                                      "--process-val-folder","val",
    #                                      "--process-eval-folder","eval"])
    print("Building processing step")
    processing_step = ProcessingStep(name="pre-processing-step",
                                     step_args=proc_args,
                                     cache_config=cache_config)
#     processing_step = ProcessingStep(name="pre-processing-step",
#                                      processor=processor,
#                                      inputs=[ProcessingInput(input_name="raw-data",
#                                                             source=input_data_path,
#                                                             destination=Join(on="",values=[PROCESSING_PATH,"/input"]))],
#                                      code=os.path.join(BASE_DIR,"processing.py"),
#                                      outputs=[ProcessingOutput(output_name=train_folder_name,
#                                                                source=Join(on="",values=[PROCESSING_PATH,"/",train_folder_name]),
#                                                                destination=Join(on="",values=[processed_data_path,"/",train_folder_name])),
#                                               ProcessingOutput(output_name=val_folder_name,
#                                                                source=Join(on="",values=[PROCESSING_PATH,"/",val_folder_name]),
#                                                                destination=Join(on="",values=[processed_data_path,"/",val_folder_name])),
#                                               ProcessingOutput(output_name=eval_folder_name,
#                                                                source=Join(on="",values=[PROCESSING_PATH,"/",eval_folder_name]),
#                                                                destination=Join(on="",values=[processed_data_path,"/",eval_folder_name]))],
                                     
#                                     cache_config=cache_config)
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
        role=role,
        source_dir=BASE_DIR,
        framework_version="1.8.0",
        py_version="py3",
        instance_count=training_instance_count,
        instance_type=training_instance_type,
        use_spot_instances=True,
        checkpoint_s3_uri=f"S3://{BUCKET_NAME}/{base_job_prefix}/checkpoints/", 
        checkpoint_local_path="/opt/ml/checkpoints/",
        max_run=3600,
        max_wait=3800,
        environment=env_variables,
        sagemaker_session=sagemaker_session)
    
    print("Setting hyperparameters")
    pt_estimator.set_hyperparameters(epochs=epochs,
                                     batch_size=batch_size)
    print("Setting training step")
    step_train = TrainingStep(
        name="training-step",
        estimator=pt_estimator,
        inputs={
            train_folder_name: TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[train_folder_name].S3Output.S3Uri
            ),
            val_folder_name: TrainingInput(
                s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[val_folder_name].S3Output.S3Uri
            )
        },
        depends_on=processing_step,
        cache_config=cache_config)
    
    # Processing step for evaluation
    print("Setting model evaluation processor")
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
    
    print("setting evaluation step args")
    eval_args = model_eval.run( inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=processing_step.properties.ProcessingOutputConfig.Outputs["eval"].S3Output.S3Uri,
                destination="/opt/ml/processing/eval"
            )
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/eval")
        ],
        code=os.path.join(BASE_DIR, "evaluation.py"))
    
    print("Setting model evaluation step")
    # step_eval = ProcessingStep(
    #     name="evaluation-step",
    #     processor=model_eval,
    #     inputs=[
    #         ProcessingInput(
    #             source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    #             destination="/opt/ml/processing/model"
    #         ),
    #         ProcessingInput(
    #             source=processing_step.properties.ProcessingOutputConfig.Outputs["eval"].S3Output.S3Uri,
    #             destination="/opt/ml/processing/eval"
    #         )
    #     ],
    #     outputs=[
    #         ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/eval")
    #     ],
    #     code=os.path.join(BASE_DIR, "evaluation.py"),
    #     property_files=[evaluation_report]
    # )
    step_eval=ProcessingStep(name="evaluation-step",
                             step_args=eval_args,
                             property_files=[evaluation_report])
    
    # Register model step that will be conditionally executed
    print("Setting model metrics")
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
    
    print("condition step")
    condition = ConditionGreaterThanOrEqualTo(left=accuracy_score,right=0.8)
    step_condition = ConditionStep(
        name="evaluation-condition-check",
        conditions=[condition],
        if_steps=[step_register],
        else_steps=[]
    )
    
    # Pipeline instance
    print("Building final pipeline")
    pipeline_session = get_session(region, default_bucket,pipe_session=True)
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
            eval_folder,
            model_approval_status,
            input_data_path,
            processed_data_path,
            model_path
        ],
        steps=[processing_step, step_train, step_eval, step_condition],
        sagemaker_session=pipeline_session
    )
    return pipeline
    
    