import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
import random
import librosa.display as dsp
from IPython.core.display_functions import display
from IPython.display import Audio
from matplotlib import ticker
from pytorch_wavelets import DWT1D
import torch_dct as dct
import warnings
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

warnings.filterwarnings("ignore", category=np.ComplexWarning)  # Suppress warning of imaginary part being discarded


### Model Training and Evaluation ###

def train_one_epoch(model, optimizer, criterion, scheduler, train_data, train_labels, batch_size, device):
    model.train()
    total_loss = 0
    total_samples = 0

    for data, labels in custom_data_loader(train_data, train_labels, batch_size):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data.unsqueeze(1))  # Pass data through model
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()  # Optimize weights based on the loss

        total_loss += loss.item() * data.size(0)  # loss multiplied by batch size
        total_samples += data.size(0)

    scheduler.step()

    return total_loss / total_samples


def evaluate(model, criterion, data, labels, batch_size, device, transformation=None):
    model.eval()
    total_loss = 0
    correct = 0
    total_samples = 0
    all_predictions = []
    true_labels = []

    with torch.no_grad():
        for data, labels in custom_data_loader(data, labels, batch_size):
            data, labels = data.to(device), labels.to(device)
            data = data.squeeze(0)
            if transformation == 'Orig':
                data = data.unsqueeze(1)
            elif transformation in ['STDFT', 'PCA', 'STDFT_PCA', 'DCT']:
                data = data.unsqueeze(0)  # ensure correct dimension
            else:
                data = reshape_tensor(data)
            outputs = model(data)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * data.size(0)  # loss multiplied by batch size
            _, predicted = torch.max(outputs, 1)  # get the index of the class with the highest probability
            correct += (predicted == labels).sum().item()
            total_samples += data.size(0)
            all_predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples

    return avg_loss, accuracy, true_labels, all_predictions


### Data Preparation ###

def custom_data_loader(data_tensor, labels_tensor, batch_size):
    dataset_size = data_tensor.shape[0]
    batches = []

    for start_idx in range(0, dataset_size, batch_size):
        end_idx = min(start_idx + batch_size, dataset_size)  # avoid out of index error
        batches.append((data_tensor[start_idx:end_idx], labels_tensor[start_idx:end_idx]))

    return batches

def pad_or_truncate_data(data, desired_length=8000, fixed_offset=False):
    if data.numel() < desired_length:  # Padding
        embedded_data = torch.zeros(desired_length, dtype=data.dtype)  # create tensor of only zeros
        if not fixed_offset:  # Randomly select starting index for embedding the audio
            max_offset = desired_length - data.numel()  # calculate maximum starting index for embedding the audio without going over 8000
            random_offset = random.randint(0, max_offset)  # Randomly select starting index within the maximum offset
        else: # if fixed_offset is True, audio is always embedded at the beginning
            random_offset = 0
        embedded_data[random_offset:random_offset + data.numel()] = data  # Embed the audio at the random offset
    elif data.numel() > desired_length:  # Truncating
        embedded_data = data[:desired_length]
    else:
        embedded_data = data.clone()

    return embedded_data


def preprocess_audio(audio, orig_sample_rate=8000, fixed_offset=False):
    # Resample the audio to 8000 Hz
    if orig_sample_rate != 8000:
        resample_transform = torchaudio.transforms.Resample(orig_freq=orig_sample_rate, new_freq=8000)
        audio_resampled = resample_transform(audio)
    else:
        audio_resampled = audio

    # Padding or truncating the audio
    embedded_data = pad_or_truncate_data(audio_resampled, fixed_offset=fixed_offset)

    # Apply percentile normalization
    audio_preprocessed = perform_percentile_normalization(embedded_data)

    return audio_preprocessed


def perform_percentile_normalization(data, percentile=0.95):
    sorted_data, _ = torch.sort(data)
    percentile_index = int(percentile * len(sorted_data)) - 1  # Calculate the index for the 95th percentile
    percentile_value = sorted_data[percentile_index]
    normalized_data = data / (percentile_value + 0.001)  # Normalize with small epsilon to avoid division by zero

    return normalized_data


def load_and_split_data(data_path, labels_path, train_size=0.7, validation_size=0.2):
    # Load the pre-processed tensors
    data_tensor = torch.load(data_path)
    labels_tensor = torch.load(labels_path)

    # Define subset sizes
    total_size = data_tensor.size(0)
    train_size = int(train_size * total_size)
    validation_size = int(validation_size * total_size)

    # Generate shuffled indices
    indices = torch.randperm(total_size).tolist()

    # Split indices for each subset
    train_indices = indices[:train_size]
    validation_indices = indices[train_size:train_size + validation_size]
    test_indices = indices[train_size + validation_size:]

    # Create subsets
    ## Training set
    train_data = data_tensor[train_indices]
    train_labels = labels_tensor[train_indices]

    ## Validation set
    validation_data = data_tensor[validation_indices]
    validation_labels = labels_tensor[validation_indices]

    ## Test set
    test_data = data_tensor[test_indices]
    test_labels = labels_tensor[test_indices]

    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def reshape_tensor(tensor):
    # Shape before being input to the first convolutional layer should be [1, 1, 8000]
    flattened_tensor = tensor.squeeze()  # Flatten

    desired_length = 8000

    tensor = pad_or_truncate_data(flattened_tensor)

    # Reshape to (1, 1, 8000)
    reshaped_tensor = tensor.view(1, 1, desired_length)
    return reshaped_tensor


def predict_instance(model, audio_tensor):
    model.eval()
    with torch.no_grad():
        prediction = model(audio_tensor).argmax().item()

    return prediction


def get_audio(digit=None, speaker=None, recording=None):
    if digit is None or digit < 0 or digit > 9:
        digit = random.randint(0, 9)

    if speaker is None or speaker < 1 or speaker > 60:
        speaker = random.randint(1, 60)
    if speaker < 10:
        speaker = f"0{speaker}"

    if recording is None or recording < 0 or recording > 49:
        recording = random.randint(0, 49)

    print(f"Digit: {digit}\nSpeaker: {speaker}\nRecording: {recording}")

    dataset_path = "AudioMNIST/data"
    audio_file_path = f"{dataset_path}/{speaker}/{digit}_{speaker}_{recording}.wav"
    print(f"Audio File Path: {audio_file_path}")

    # Get Audio from the location
    audio, sample_rate = torchaudio.load(audio_file_path)

    return audio, sample_rate, audio_file_path, digit


def get_layer_output(model, layer, audio_tensor_reshaped):
    # Hook to capture the layer output
    def hook(module, input, output):
        return output

    handle = layer.register_forward_hook(hook)

    # Forward pass to get the output
    with torch.no_grad():
        model(audio_tensor_reshaped)

    handle.remove()

    return hook.output


def perform_STDFT(data, inverse=False):
    # Define Spectogram parameters

    n_fft = 2048
    win_length = n_fft
    hop_length = n_fft // 4
    window = torch.hann_window

    if inverse:
        inverse_spectogram = torchaudio.transforms.GriffinLim(n_fft=n_fft, win_length=win_length,
                                                              hop_length=hop_length,
                                                              window_fn=window)  # Compute waveform from a linear scale magnitude spectrogram
        data = inverse_spectogram(data)
    else:
        spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length,
                                                                  hop_length=hop_length, window_fn=window,
                                                                  power=2.0)  # Compute power spectogram -> .abs().pow(2)
        data = spectrogram_transform(data)

    return data


### Data Plotting ###

def prepare_for_plotting(data, magnitudes=False, average=False, scale=False):
    # Tensor to Numpy for plotting
    data_np = data.detach().cpu().numpy().squeeze().astype(np.float64)

    if average:
        data_np = np.mean(data_np, axis=0)  # Average the outputs (filters) of the Conv1D layer
    if magnitudes:
        data_np = np.abs(data_np)
    if scale and not magnitudes:
        abs_max = np.max(np.abs(data_np))
        if abs_max == 0:
            data_np = np.zeros(data_np.shape)
        else:
            data_np = data_np / abs_max
    elif scale and magnitudes:
        scaler = MinMaxScaler()
        data_np = scaler.fit_transform(data_np.reshape(-1, 1))

    return data_np


def plot_and_play(title, audio, sample_rate=8000):
    # Plot the audio wave
    plt.figure()
    dsp.waveshow(audio, sr=sample_rate, color='gray')
    plt.title(title, fontsize=27)
    plt.xlabel('Time [s]', fontsize=20)
    plt.ylabel('Amplitude', fontsize=20)

    # Make plot better readable for user study
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=18)  # Adjust tick label sizes
    formatter = ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x))  # Limit x-axis to 1 decimal places
    ax.xaxis.set_major_formatter(formatter)

    plt.show()

    # Show the widget (playable audio)
    display(Audio(data=audio, rate=sample_rate))


def create_confusion_matrix(true_labels, predictions, classes, transformation):
    if transformation == 'Conv':
        transformation = 'original'

    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted digits', fontsize=16)
    plt.ylabel('True digits', fontsize=16)
    plt.title(f'Confusion Matrix for {transformation} Model', fontsize=25)
    plt.plot()


def define_plot_range(start_index, end_index, data):
    n = len(data)

    if start_index < 0:
        start_index = 0
        print("Start index is less than 0. Defaulting to 0.")

    if end_index > n:
        end_index = n
        print(f"End index is beyond the maximum of {n}. Resetting to the maximum.")

    if end_index <= start_index or (end_index - start_index) < 100:
        end_index = min(start_index + 100, n)  # Avoid going out of bounds
        print(f"End index must be at least 100 larger than start index. Adjusting end index to {end_index}.")

    range = np.arange(start_index, end_index)

    return start_index, end_index, range


def create_title(magnitudes=False, digit=None, average=False, scale=False):
    title_parts = []
    if magnitudes:
        title_parts.append(" Magnitudes")
    if digit is not None:
        title_parts.append(f" for Digit {digit}")
    if average and not scale:
        title_parts.append(" (averaged)")
    elif scale and not average:
        title_parts.append(" (scaled)")
    elif scale and average:
        title_parts.append(" (averaged and scaled)")
    if not title_parts:
        return ""  # Return empty string if none are True
    return "".join(title_parts)


def adjust_plot(label_size=18):
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=label_size)
    ax.set_aspect('auto', adjustable='box')
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    plt.show()


def setup_plot(title, x_label, y_label):
    plt.figure(figsize=(12, 6))
    plt.title(title, fontsize=22)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)


def display_plot(data, transformation, label=None, range=None, color='gray', cmap='viridis'):
    if transformation == 'STDFT':
        plt.imshow(data, aspect='auto', origin='lower', cmap=cmap, extent=[0, 1, 0, data.shape[0]])
        cbar = plt.colorbar(format="%+2.0f")
        cbar.set_label(label, fontsize=18)
        cbar.ax.tick_params(labelsize=18)  # Adjust the tick label size on the colorbar
    else:
        # Change sample index to time for Conv and DWT
        if transformation == 'Conv':
            stop = len(data) / 8000
            range = np.linspace(0, stop, num=len(data))  # Generate x values scaled from 0 to 1
        elif transformation == 'DWT':
            stop = len(data) / 1004
            range = np.linspace(0, stop, num=len(data))

        if range is not None:
            plt.plot(range, data, color=color)
        else:
            plt.plot(data, color=color)
    adjust_plot()


def plot_data(transformation, data, digit, scale=False, magnitudes=False, start_index=0, end_index=8000):
    average = False
    range = 8000

    # Prepare Data
    if transformation == 'Conv':
        x_label = "Time [s]"
        y_label = "Activation Value"
        average = True

    elif transformation == 'DFT':
        data = torch.fft.fft(data)

        x_label = "Frequency [Hz]"
        y_label = "Magnitudes"

    elif transformation == 'STDFT':
        data = perform_STDFT(data)

        x_label = "Time [s]"
        y_label = "Frequency [Hz]"

        scale = False
        magnitudes = False

    elif transformation == 'DCT':
        data = dct.dct(data, norm='ortho')

        x_label = "Frequency [Hz]"
        y_label = "Magnitudes"

    elif transformation == 'DWT':
        xfm = DWT1D(J=3, mode='zero', wave='db3')

        data = reshape_tensor(data)  # DWT requires 3D input
        data, Yh = xfm(data)
        data = data[0].squeeze()  # Approximation Coefficients

        x_label = 'Time [s] (downsampled)'
        y_label = 'Coefficient Value'

    else:
        print("Unavailable transformation. Please choose from 'Conv', 'DFT', 'STDFT', 'DCT', or 'DWT'.")

    data = prepare_for_plotting(data=data, magnitudes=magnitudes, average=average, scale=scale)

    # Custom indexing
    if transformation != "STDFT":
        start_index, end_index, range = define_plot_range(start_index, end_index, data)
        data = data[start_index:end_index]

    # Plotting
    if transformation == 'DWT':
        title = 'DWT Approximation Coefficients' + create_title(magnitudes=magnitudes, digit=digit, scale=scale)

        setup_plot(title=title, x_label=x_label, y_label=y_label)
        display_plot(data=data, transformation=transformation, range=range, color='gray')

        """
        # Optional: Plotting each level of Detail Coefficients separately (Relevancy only implemented for Approximation Coefficients yet)
        for i, yh in enumerate(Yh, 1):
            title = f"Detail Coefficients Level {i}" + create_title(magnitudes=magnitudes, digit=digit, scale=scale)
            Yh = prepare_for_plotting(yh[0].squeeze(), magnitudes=magnitudes, average=average, scale=scale)

            setup_plot(title=title, x_label=x_label, y_label=y_label)
            display_plot(data=Yh, plot_type=plot_type, color='gray')
        """

    else:
        title = transformation + create_title(magnitudes=magnitudes, digit=digit, average=average, scale=scale)

        setup_plot(title=title, x_label=x_label, y_label=y_label)

        if transformation == 'STDFT':
            display_plot(data=np.log2(data + 1e-6), label='Decibels (log2)', transformation=transformation)

        else:
            display_plot(data=data, transformation=transformation, range=range, color='gray')


def plot_relevancy(transformation, attr, digit, average=False, start_index=0, end_index=8000):
    scale = True
    range = 8000

    # Prepare Relevancy
    if transformation == 'Conv':
        x_label = 'Time [s]'
        average = True

    elif transformation == 'STDFT':
        scale = False

    elif transformation in ['DFT', 'DCT']:
        x_label = "Frequency [Hz]"

    elif transformation == 'DWT':
        x_label = 'Time [s] (downsampled)'
        attr = attr[:, :, :1004]

    relevancy_prepared = prepare_for_plotting(data=attr, average=average, scale=scale)

    if transformation != 'STDFT':
        start_index, end_index, range = define_plot_range(start_index, end_index, relevancy_prepared)
        relevancy_prepared = relevancy_prepared[start_index:end_index]

        if transformation == 'Conv':
            stop = len(relevancy_prepared) / 8000
            range = np.linspace(0, stop, num=len(relevancy_prepared))  # Generate x values scaled from 0 to 1
        elif transformation == 'DWT':
            stop = len(relevancy_prepared) / 1004
            range = np.linspace(0, stop, num=len(relevancy_prepared))

    # Plot Relevancy
    plt.figure(figsize=(12, 6))

    if transformation == 'STDFT':
        title = f"{transformation} Relevancy" + create_title(digit=digit, average=average, scale=True)

        setup_plot(title=title, x_label='Time [s]', y_label='Frequency [Hz]')
        display_plot(data=np.log2(relevancy_prepared + 1e-6), label='Relevancy (log2)', transformation=transformation,
                     range=range, cmap='RdBu_r')

    else:
        if transformation == 'DWT':
            title = f"DWT Approximation Coefficients Relevancy" + create_title(digit=digit, average=average,
                                                                               scale=scale)
        else:
            title = f"{transformation} Relevancy" + create_title(digit=digit, average=average, scale=scale)
        plt.title(title, fontsize=22)

        plt.fill_between(range, relevancy_prepared, where=(relevancy_prepared > 0), color='red', interpolate=True)
        plt.fill_between(range, relevancy_prepared, where=(relevancy_prepared <= 0), color='blue', interpolate=True)
        y_label = 'Relevancy'

    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)

    adjust_plot()
