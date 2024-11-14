from easy_tpp.config_factory import Config
from src.tpprunner import TPPRunner
from easy_tpp.utils import logger
import plotly.graph_objects as go
from pathlib import Path
import yaml, pickle, json
import numpy as np


def evaluate_model(args):

    # read configuration from args.config
    dataset_config_path = Path(args.source) / "training_datasets" / args.name / 'config.json'
    with open(dataset_config_path, 'r') as f:
        dataset_config = json.load(f)

    model_path = Path(args.source) / "training_results" / args.name / args.id
    yaml_file = next(model_path.glob("*.yaml"))
    with open(yaml_file, 'r') as file:
        training_output_config = yaml.load(file, Loader=yaml.FullLoader)

    # fix the base_dir for the generation stage
    training_base_dir = training_output_config['base_config']['base_dir']
    evaluation_base_dir = training_base_dir.replace("training_results", "evaluation_results")

    experiment_id = f"{training_output_config['base_config']['model_id']}_eval"
    # Transform the dict to match training configuration format
    config = {
        "pipeline_config_id": "runner_config",
        "data": {
            training_output_config['base_config']['dataset_id']: {
                "data_format": training_output_config['data_config']['data_format'],
                "train_dir": training_output_config['data_config']['train_dir'],
                "valid_dir": training_output_config['data_config']['valid_dir'],
                "test_dir": training_output_config['data_config']['test_dir'],
                "data_specs": {
                    "num_event_types": training_output_config['data_config']['data_specs']['num_event_types'],
                    "pad_token_id": training_output_config['data_config']['data_specs']['pad_token_id'],
                    "padding_side": training_output_config['data_config']['data_specs']['padding_side'],
                    "truncation_side": training_output_config['data_config']['data_specs']['truncation_side'],
                    "padding_strategy" : training_output_config['data_config']['data_specs']['padding_strategy'],
                }
            }
        },
        experiment_id: {
            "base_config": {
                "stage": "eval",
                "backend": training_output_config['base_config']['backend'],
                "dataset_id": training_output_config['base_config']['dataset_id'],
                "runner_id": training_output_config['base_config']['runner_id'],
                "model_id": training_output_config['base_config']['model_id'],
                "base_dir": evaluation_base_dir,
            },
            "trainer_config": {
                "batch_size": training_output_config['trainer_config']['batch_size'],
                "max_epoch": training_output_config['trainer_config']['max_epoch'],
                "shuffle": training_output_config['trainer_config']['shuffle'],
                "optimizer": training_output_config['trainer_config']['optimizer'],
                "learning_rate": training_output_config['trainer_config']['learning_rate'],
                "valid_freq": training_output_config['trainer_config']['valid_freq'],
                "use_tfb": training_output_config['trainer_config']['use_tfb'],
                "metrics": training_output_config['trainer_config']['metrics'],
                "seed": training_output_config['trainer_config']['seed'],
                "gpu": -1,#training_output_config['trainer_config']['gpu'],
            },
            "model_config": {
                "model_specs" : training_output_config['model_config']['model_specs'],
                "hidden_size": training_output_config['model_config']['hidden_size'],
                "num_layers": training_output_config['model_config']['num_layers'],
                "loss_integral_num_sample_per_step": training_output_config['model_config']['loss_integral_num_sample_per_step'],
                "use_ln": training_output_config['model_config']['use_ln'],
                "pretrained_model_dir": training_output_config['base_config']['specs']['saved_model_dir'],
                "thinning": training_output_config['model_config']['thinning'],
                "noise_regularization": training_output_config['model_config']['noise_regularization'] if 'noise_regularization' in training_output_config['model_config'] else {} 
            }
        }
    }
    config = Config.build_from_dict(config, experiment_id=experiment_id)
    model_runner = TPPRunner(config)
    model_runner.run()


def plot_predictions(args):

    # read configuration from args.config
    dataset_config_path = Path(args.source) / "training_datasets" / args.name / 'config.json'
    with open(dataset_config_path, 'r') as f:
        dataset_config = json.load(f)
    
    model_path = Path(args.source) / "prediction_results" / args.name / args.id
    yaml_file = next(model_path.glob("*.yaml"))
    with open(yaml_file, 'r') as file:
        generation_output_config = yaml.load(file, Loader=yaml.FullLoader)
    
    pkl_file = next(model_path.glob("*.pkl"))
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)

    model_id = generation_output_config['base_config']['model_id']
    if model_id == 'IntensityFree2D' and generation_output_config['prediction_config']['method'] == 'probabilistic':
        plot_probability_predictions_2D(dataset_config, generation_output_config, data, model_path, args)
    elif model_id == 'IntensityFree' and generation_output_config['prediction_config']['method'] == 'probabilistic':
        plot_probability_predictions_1D(dataset_config, generation_output_config, data, model_path, args)

def plot_probability_predictions_1D(dataset_config, generation_output_config, data, model_path, args):

    dtime_data = []
    event_type_data = []
    for batch in data['pred']:
        dtime_data.append(batch[0])
        event_type_data.append(batch[1])
    concatenated_label_dtime = np.concatenate(dtime_data, axis=0)
    concatenated_label_event_type = np.concatenate(event_type_data, axis=0)

    max_index = concatenated_label_dtime.shape[0]
    ar_index = np.random.randint(0, max_index, size=1)[0]
    assert ar_index < max_index, f"Index out of range: {ar_index} > {max_index}"
    dtime_logprob_pred = concatenated_label_dtime[ar_index,:]
    dtime_prob_pred = np.exp(dtime_logprob_pred)

    # dtime samples
    prediction_config = generation_output_config['prediction_config']
    sample_dtime_min = prediction_config['probability_generation']['sample_dtime_min']
    sample_dtime_max = prediction_config['probability_generation']['sample_dtime_max']
    num_steps_dtime = prediction_config['probability_generation']['num_steps_dtime']
    dtime_samples = np.linspace(sample_dtime_min, sample_dtime_max, num_steps_dtime)
    
    # Create a scatter plot using plotly
    fig = go.Figure(data=go.Scatter(x=dtime_samples, y=dtime_prob_pred, mode='markers'))
    fig.update_layout(title='Predictions', xaxis_title='Time', yaxis_title='Probability')
    fig.write_html(model_path / "prob_delta_times.html")

    event_types_length = concatenated_label_event_type.shape[1] # event_types length
    event_types_probs = np.exp(concatenated_label_event_type[ar_index,:])


    # Define class labels
    class_labels = [f"Event type {i}" for i in range(event_types_length)]

    # Create a figure
    fig = go.Figure()

    # Add traces for each set of probabilities
    fig.add_trace(
        go.Bar(
            x=class_labels,
            y=event_types_probs,
            opacity=0.7
        )
    )
    # Update layout for better aesthetics
    fig.update_layout(
        title="Probability Distribution Across Classes",
        xaxis_title="Class Number",
        yaxis_title="Probability",
        barmode='group',  # Options: 'group', 'overlay', 'stack'
        template='plotly_white',
        legend_title="Probability Sets",
        xaxis_tickangle=-45
    )
    fig.write_html(model_path / "prob_event_types.html")

    # Optionally, adjust y-axis to range [0, 1] if dealing with probabilities
    #fig.update_yaxes(range=[0, 1])
    # calculate the rmse of dtime
    #concatenated_label_dtime = np.concatenate(data['label'], axis=1)
    #label_dtime = concatenated_label_dtime[1, :, -1]
    #concatenated_pred_dtime_mean = np.concatenate(data['dtime_mean_pred'], axis=0)
    #pred_dtime = concatenated_pred_dtime_mean[:, 0]
    #rmse = np.sqrt(np.mean((pred_dtime - label_dtime) ** 2))
    #logger.info(f"RMSE: {rmse}")

    # calculate the average loglikelihood
    #loglikelihood = np.array(data['loglikelihood']).sum()/len(pred_dtime)
    #logger.info(f"loglikelihood: {loglikelihood}")


def plot_probability_predictions_2D(dataset_config, generation_output_config, data, model_path, args):

    # (num_batches, 2, num_steps*num_steps, batch_size, seq_len-2)
    concatenated_label_dtime = np.concatenate(data['pred'], axis=2)[0,...]
    # (num_steps*num_steps, batch_size, seq_len-2)
    max_index = concatenated_label_dtime.shape[0]
    ar_index = np.random.randint(0, max_index, size=1)
    assert ar_index < max_index, f"Index out of range: {ar_index} > {max_index}"
    dtime_logcdf_pred = concatenated_label_dtime[:,ar_index,-1]
    joint_pdf_pred = np.exp(dtime_logcdf_pred)

    prediction_config = generation_output_config['prediction_config']
    sample_dtime_min = prediction_config['probability_generation']['sample_dtime_min']
    sample_dtime_max = prediction_config['probability_generation']['sample_dtime_max']
    num_steps_dtime = prediction_config['probability_generation']['num_steps_dtime']
    sample_event_type_min = prediction_config['probability_generation']['sample_event_type_min']
    sample_event_type_max = prediction_config['probability_generation']['sample_event_type_max']
    num_steps_event_type = prediction_config['probability_generation']['num_steps_event_type']
    # dtime samples, for all batches and position in seq is the same
    # Step 1: Create 1D arrays for time_since_last_event and event_types
    time_since_last_event = np.linspace(sample_dtime_min, sample_dtime_max, num_steps_dtime)
    event_types = np.linspace(sample_event_type_min, sample_event_type_max, num_steps_event_type)

    # Step 2: Generate a 2D meshgrid with 'ij' indexing to match PyTorch's behavior
    time_grid, event_grid = np.meshgrid(time_since_last_event, event_types, indexing='ij')

    # Step 3: Stack the grids along the last axis to create a grid of 2D samples
    # Shape: (num_steps_dtime, num_steps_event_type, 2)
    sample_grid = np.stack((time_grid, event_grid), axis=-1)

    # Step 4: Reshape to [num_samples * num_samples, 2]
    # where num_samples = num_steps_dtime * num_steps_event_type
    dtime_samples = sample_grid.reshape(-1, 2)

    
    # Create a scatter plot using plotly
    # Create a 3D surface plot with Plotly
    grid_size = int(np.sqrt(joint_pdf_pred.shape[0]))
    # cap values of joint_pdf_pred to 2
    joint_pdf_pred[joint_pdf_pred > 2] = 2
    # plot
    Z = joint_pdf_pred.reshape(grid_size, grid_size)  # Probability values
    X = dtime_samples[:, 0].reshape(grid_size, grid_size)  # X-axis (inter_times)
    Y = dtime_samples[:, 1].reshape(grid_size, grid_size)  # Y-axis (event_types)
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(
        title="3D Probability Density Plot",
        scene=dict(
            xaxis_title="Inter Times (X)",
            yaxis_title="Event Types (Y)",
            zaxis_title="Probability (Z)"
        )
    )
    fig.update_layout(title='Predictions', xaxis_title='Time', yaxis_title='Probability')
    #fig.update_layout(scene_zaxis_range=[0, 2]) # limit z axis
    fig.write_html(model_path / "prob_joint_3d.html")
    
    

    # calculate the rmse of dtime
    #concatenated_label_dtime = np.concatenate(data['label'], axis=1)
    #label_dtime = concatenated_label_dtime[1, :, -1]
    #concatenated_pred_dtime_mean = np.concatenate(data['dtime_mean_pred'], axis=0)
    #pred_dtime = concatenated_pred_dtime_mean[:, 0]
    #rmse = np.sqrt(np.mean((pred_dtime - label_dtime) ** 2))
    #logger.info(f"dtime RMSE: {rmse}")

    # calculate the rmse of event types
    #label_event_types = concatenated_label_dtime[2, :, -1]
    #concatenated_pred_type_mean = np.concatenate(data['type_mean_pred'], axis=0)
    #pred_type = concatenated_pred_type_mean[:, 0]
    #rmse = np.sqrt(np.mean((pred_type - label_event_types) ** 2))
    #logger.info(f"type RMSE: {rmse}")

    # calculate the average loglikelihood (over the batches)
    #loglikelihood = np.array(data['loglikelihood']).sum()/len(data['loglikelihood'])
    #logger.info(f"loglikelihood: {loglikelihood}")
