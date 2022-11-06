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
    :param y_true:                   Actual y_true data support types (mlrun.DataItem, list, dict, pd.DataFrame, pd.Series, np.ndarray)
    :param metrics:                  list of all the metrics that you want to be calculate - each result will save with name format
                                     {model_name}_{model_tag or None}_{metrics_name}.
    :param batch_id:                 The ID of the current batch (inference dataset) to compare to the y_true.
    :param switch_model:             Default False, if True automatic change the label for the model with the best average result for the selected metric
    :param log_result:               Default True, log the result to the run.spec
    :param comparsion_id:            The ID of the given compare run uses to compare to other prior run. If `None`, it will be generated for future use.
                                     Will be logged as a result of the run.
    :param label_columns:            The target label(s) of the column(s) in the dataset for Regression or
                                     Classification tasks. The label column can be accessed from the model object, or
                                     the feature vector provided if available.
    :param last_run_avg:             Number of the last runs you want to compare to. default = 0, for exmaple : 0 for compare all previous runs, 4 compare to last 4 runs.
    :param champion_metric:          According to this metric the function selects the champion (example: champion_metric : accuracy, means that the
                                     champion will be selected after a comparision between the accuracy metric result.
    :param artifacts_tag:            Tag to use for all the artifacts resulted from the function.
    :param tag:                      The model's tag to log with.
    :param weights:                  Dict of metric name and a float to multiply the avg result with {'accuracy' : 0.6,'f1_score' : 0.8}
    :param minimum_champion_score:   Minimum score to select new model champion from, for example: if minimum_champion_score = 0.8, and the max result is 0.7 it will not
                                    selecte a newchampion
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

            elif metric == 'f1_score':
                result = round(f1_score(y_true, y_pred), 3)

            results['Results'][model_name + f'_{metric}'] = round(result, 3) # Save metric score
            if model_name not in results_df:
                results_df[model_name + f'_{metric}'] = {f'Score_{str(date.today())}_{datetime.now().strftime("%H:%M:%S")}' : result}

            else:
                results_df[model_name + f'_{metric}'].update({f'Score_{str(date.today())}_{datetime.now().strftime("%H:%M:%S")}' : result})
            context.logger.info(f"{model_name + f'_{metric}'}---->{result}")
    df_result_runs = pd.DataFrame(results_df).transpose().fillna(0)
    context.log_result(key='Batch_ID',value=batch_id)

    if comparsion_id is None:
        compare = False
        comparsion_id = hashlib.sha224(str(datetime.now()).encode()).hexdigest()

    context.logger.info(f"Comparsion ID--->{comparsion_id}")
    key = f'Result_{comparsion_id[:4]}'
    context.log_dataset(key=key,df=df_result_runs,format='csv',db_key=key,tag=artifacts_tag)


    # Query all runs with the same compare ID
    if compare:
        run_objs = mlrun.get_run_db().list_runs(labels=[f'comparsion_id={comparsion_id}']).to_objects()
        count = len(run_objs)
        if last_run_avg:
            start = len(run_objs) - last_run_avg
        for run_obj in run_objs[start:]:
            run_results = run_obj.status.results['Results']
            for model_name in run_results:

                if "AVG" not in model_name:
                    if model_name + "_AVG" not in models_score:
                        models_score[model_name + "_AVG"] = [run_results[model_name]]
                    else:
                        models_score[model_name + "_AVG"].append(run_results[model_name])
        count += 1
    for model_name in results['Results']:
        if model_name + "_AVG" not in models_score:
            models_score[model_name + "_AVG"] = [results['Results'][model_name]]
        else:
            models_score[model_name + "_AVG"].append(results['Results'][model_name])

    # Calculate Average
    for model_name in models_score:
        results['Results'][model_name] = round((sum(models_score[model_name]))/len(models_score[model_name]),3)

    #Create ad DataFrame of the AVG score (Index = model_name_tag , columns = metrics)
    df_dict = {}
    for metric in metrics:
        for model_name in models_score:
            if metric in model_name :
                _model_name = model_name.replace(f"_{metric}_AVG","")
                if _model_name in df_dict:
                    if df_dict[_model_name] != {}:
                        df_dict[_model_name].update({metric : models_score[model_name][0]})
                else:
                    df_dict[_model_name]={metric : models_score[model_name][0]}

    df_result = pd.DataFrame(df_dict).transpose().fillna(0)

    # Multiply the AVG score with the relvant metric
    if  weights != {}:
        weights = _check_weights(weights,metrics)
        df_result_weights=df_result.mul(weights).copy()
        df_result['accuracy_result'] = df_result['accuracy']
        df_result['f1_score_result'] = df_result['f1_score']
        df_result['accuracy'] = df_result_weights['accuracy']
        df_result['f1_score'] = df_result_weights['f1_score']


    context.log_results(results=results)
    fig = go.Figure()
    df_champion_metric = df_result[champion_metric].copy()
    for champion_name,champion_score in df_result[df_result[champion_metric]==df_champion_metric.max()][champion_metric].items():
        champion_name = champion_name
        champion_score = champion_score


    for metric in metrics:
        for model_name in models_score:
            if metric in model_name and model_name[:-4] == champion_name:
                fig.add_trace(
                    go.Scatter(x=list(range(len(models_score[model_name]))), y=models_score[model_name], name=f"{model_name} (Champion)",
                               line_shape='linear'))
            elif metric in model_name :
                fig.add_trace(go.Scatter(x=list(range(len(models_score[model_name]))), y=models_score[model_name], name=model_name,
                                         line_shape='linear'))
        artifact = PlotlyArtifact(key=f"comparsion_{metric}_{comparsion_id[:4]}.html", figure=fig)
        context.log_artifact(artifact, tag=artifacts_tag,db_key=f"comparsion_{metric}_{comparsion_id[:4]}.html")
        fig = go.Figure()


    if log_result:  # pkl
        context.logger.info(f"Logging results")
        context.log_results(results=results)
        context.log_dataset(f'Results_Compare_{comparsion_id[:4]}',df=df_result,format='csv',db_key=f'Results_Compare_{comparsion_id[:4]}',tag=artifacts_tag)

    if switch_model:
        if champion_score < minimum_champion_score:
            context.logger.info(f'Champion results {champion_score} is lower then the Minumun score, Champion will not be Selected')

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
