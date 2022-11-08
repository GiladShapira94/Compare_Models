import mlrun
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from typing import Union
from mlrun.datastore import StoreManager
import hashlib
from datetime import datetime
from datetime import date
from mlrun.artifacts import PlotlyArtifact
import plotly.graph_objects as go
import numpy as np

DatasetType = Union[mlrun.DataItem, list, dict, pd.DataFrame, pd.Series, np.ndarray]

def _check_weights(weights,metrics):
    for metric in metrics:
        if metric not in weights:
            weights[metric] = 1

    return weights
def _model_get_obj(run_obj, model_objs):
    model_uri = run_obj.spec.parameters['model']
    model_obj = model = StoreManager().get_store_artifact(model_uri)[0]  # Save the model obj for later use
    model_objs.append(model_obj)

    return model_obj, model_objs


def compare_models(y_true: DatasetType,
                   context=mlrun.MLClientCtx,
                   metrics: list = [ ],
                   batch_id: str = '',
                   switch_model: bool = False,
                   log_result: bool = True,
                   comparsion_id: str = None,
                   label_columns='',
                   artifacts_tag: str = "",
                   last_run_avg : int = 0,
                   champion_metric : str = '',
                   tag: str = "" ,
                   weights : dict = {},
                   minimum_champion_score : str = 0 ):
    """
    Perform a comparison between models base on their batch prediction (compare thier y_pred to the actual y_true).
    Calculate for each model in a batch predict result for each metrics and selecting the best model by the maximum average result.
    Can compare between previus runs by Comparison ID and calculate the average result for each model

    :param context:                  MLRun context.
    :param y_true:                   Actual y_true support type (mlrun.DataItem, list, dict, pd.DataFrame, pd.Series, np.ndarray)
    :param metrics:                  list of all the metrics that you want to be calculate - each result will save with name format
                                     {model_name}_{model_tag or None}_{metrics_name}
    :param batch_id:                 The ID of the given batch (inference dataset) to comapre.
    :param switch_model:             Default False, if True automatic change the label for the model with the best average result for the selected champion_metric
    :param log_result:               Default True, log the result to the run.spec
    :param comparsion_id:            The ID of the given compare run uses to compare to other run. If `None`, it will be generated.
                                     Will be logged as a result of the run.
    :param batch_id:                 The ID of the given batch (inference dataset). If `None`, it will be generated.
                                     Will be logged as a result of the run.
    :param label_columns:            The target label(s) of the column(s) in the dataset for Regression or
                                     Classification tasks. The label column can be accessed from the model object, or
                                     the feature vector provided if available.
    :param last_run_avg:             Number of the last run you want to compare to. default = 0 (compare to all previus runs)
    :param champion_metric:          According to this metric the function compare the result and selecting the champion (example: champion_metric : accuracy, means that the
                                     champion will be selected after a comparision between accuracy metric result.
    :param artifacts_tag:            Tag to use for all the artifacts resulted from the function.
    :param tag:                     The model's tag to log with
    """

    run_objs = mlrun.get_run_db().list_runs(labels=[f'batch_id={batch_id}']).to_objects()
    results = {"Results":{}}
    results_df = {}
    champion_name = ''
    champion_result = 0
    model_objs = []
    compare = True
    models_score = {}
    count = 1
    start = last_run_avg
    if isinstance(y_true, mlrun.DataItem):
        y_true_df = y_true.as_df()
        y_true = pd.DataFrame(y_true_df)

    else:
        y_true = y_true

    context.logger.info(f"Starting compare models Batch ID {batch_id}")

    # Calculate result for a batch inference run
    start_time = f'{str(date.today())}_{datetime.now().strftime("%H:%M:%S")}'
    for run_obj in run_objs:

        model_obj, model_objs = _model_get_obj(run_obj, model_objs)
        predict_file_name = run_obj.spec.parameters.get('result_set_name', 'prediction')
        predict_store_uri = run_obj.outputs[predict_file_name]
        model_tag =  "_" + model_obj.tag if model_obj.tag else ""
        model_name = model_obj.db_key + model_tag

        # Select only the y_pred column
        y_pred = mlrun.get_dataitem(predict_store_uri).as_df()[label_columns]

        for metric in metrics:
            if metric == 'accuracy':
                result = round(accuracy_score(y_true, y_pred), 3)
                results_df[model_name + f'_{metric}'] = {f'Score_{start_time}' : result}

            elif metric == 'f1_score':
                result = round(f1_score(y_true, y_pred), 3)
                results_df[model_name + f'_{metric}'] = {f'Score_{start_time}' : result}

            results['Results'][model_name + f'_{metric}'] = round(result, 3) # Save metric score
            context.logger.info(f"{model_name + f'_{metric}'}---->{result}")

    context.log_result(key='Batch_ID',value=batch_id)

    if comparsion_id is None:
        compare = False
        comparsion_id = hashlib.sha224(str(datetime.now()).encode()).hexdigest()

    context.logger.info(f"Comparsion ID--->{comparsion_id}")
    key = f'Result_{comparsion_id[:4]}'


    # Query all runs with the same compare ID
    if compare:
        run_objs = mlrun.get_run_db().list_runs(labels=[f'comparsion_id={comparsion_id}']).to_objects()
        count = len(run_objs)
        if last_run_avg:
            start = len(run_objs) - last_run_avg

        for run_obj in run_objs[start:]:
            run_results = run_obj.status.results['Results']
            for model_name in run_results:
                if model_name not in results_df:
                    results_df[model_name] = {f"Score_{run_obj.status.start_time.split('.')[0].replace('T','_')}" : round(run_results[model_name],3)}

                else:
                    results_df[model_name].update({f"Score_{run_obj.status.start_time.split('.')[0].replace('T','_')}" : round(run_results[model_name],3)})


        count += 1

    df_result_runs = pd.DataFrame(results_df).transpose()
    context.log_dataset(key=key,df=df_result_runs,format='csv',db_key=key,tag=artifacts_tag)
    for metric in metrics:
        for model_name in df_result_runs.index:
            if metric in model_name:
                _model_name = model_name.replace(f"_{metric}","")
                avg_score = round(df_result_runs.loc[model_name,:].mean(),3)
                if _model_name not in models_score:
                    models_score[_model_name]= {f"{metric}" : avg_score}
                else:
                    models_score[_model_name].update({f"{metric}" : avg_score})

    #Create ad DataFrame of the AVG score (Index = model_name_tag , columns = metrics
    df_result = pd.DataFrame(models_score).transpose()
    # Multiply the AVG score with the relvant metric

    if  weights != {}:
        weights = _check_weights(weights,metrics)
        df_result_weights=df_result.mul(weights).copy()
        df_result['accuracy_result_AVG'] = df_result['accuracy']
        df_result['f1_score_result_AVG'] = df_result['f1_score']
        df_result['accuracy_Weight_AVG'] = df_result_weights['accuracy'].round(3)
        df_result['f1_score_Weight_AVG'] = df_result_weights['f1_score'].round(3)

    else:
        df_result['accuracy_result_AVG'] = df_result['accuracy']
        df_result['f1_score_result_AVG'] = df_result['f1_score']





    context.log_results(results=results)
    fig = go.Figure()
    df_champion_metric = df_result[champion_metric].copy()
    for champion_name,champion_score in df_result[df_result[champion_metric]==df_champion_metric.max()][champion_metric].items():
        champion_name = champion_name + " (Champion)"
        champion_score = champion_score

    df_result=df_result.drop(['accuracy','f1_score'],axis=1)
    context.log_dataset(f'Results_Compare_{comparsion_id[:4]}',df=df_result,format='csv',db_key=f'Results_Compare_{comparsion_id[:4]}',
                        tag=artifacts_tag)
    for metric in metrics:
        for model_name in df_result_runs.index:
            _model_name = model_name.replace(f"_{metric}","")
            if metric in model_name and _model_name in champion_name:
                fig.add_trace(
                    go.Scatter(x=list(range(len(df_result_runs.loc[model_name,:]))), y=list(df_result_runs.loc[model_name,:])[::-1], name=champion_name,
                               line_shape='linear'))
            elif metric in model_name :
                fig.add_trace(go.Scatter(x=list(range(len(df_result_runs.loc[model_name,:]))), y=list(df_result_runs.loc[model_name,:])[::-1], name=model_name,
                                         line_shape='linear'))
        artifact = PlotlyArtifact(key=f"comparsion_{metric}_{comparsion_id[:4]}.html", figure=fig)
        context.log_artifact(artifact, tag=artifacts_tag,db_key=f"comparsion_{metric}_{comparsion_id[:4]}.html")
        fig = go.Figure()


    if log_result:  # pkl
        context.logger.info(f"Logging results")
        context.log_results(results=results)

    if switch_model:
        if champion_score < minimum_champion_score:
            context.logger.info(f'Champion results {champion_score} is lower then the Minumun score, Champion will not be Switch')

        else:
            context.logger.info(f"Update {champion_name} to Status Champion")
            for model_obj in model_objs:
                model_tag = "_" + model_obj.tag if model_obj.tag else ""
                model_name = model_obj.db_key + model_tag
                if model_name in champion_name:  # mlrun.artifact - update model
                    model_obj.labels['Status'] = 'Champion'
                    mlrun.artifacts.update_model(model_obj, labels={'Status': 'Champion'})
                else:
                    model_obj.labels['Status'] = 'Challenger'
                    mlrun.artifacts.update_model(model_obj, labels={'Status': 'Challenger'})
                context.log_result(key=f'{champion_name} Champion',value=champion_result)
    context.set_label(key='comparsion_id',value=comparsion_id)
