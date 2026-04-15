from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SEED = 42
DEFAULT_TRAINING_SCHEDULE = (
    (5e-3, 400),
    (2e-3, 800),
    (1e-3, 1200),
    (1e-3, 2000),
    (5e-4, 3000),
    (1e-4, 3000),
    (5e-5, 4000),
    (1e-5, 5000),
)


@dataclass(frozen=True)
class DatasetArtifacts:
    csi_freq_domain_train: np.ndarray
    groundtruth_positions_train: np.ndarray
    groundtruth_floor_indices: np.ndarray | None = None
    array_positions_3d: np.ndarray | None = None
    center_positions_3d: np.ndarray | None = None

    @property
    def floor_nr(self) -> int:
        if self.array_positions_3d is not None:
            return int(self.array_positions_3d.shape[0])
        if self.groundtruth_floor_indices is not None:
            return int(np.unique(self.groundtruth_floor_indices).shape[0])
        return int(np.unique(self.groundtruth_positions_train[:, -1]).shape[0])

    @property
    def ue_heights(self) -> np.ndarray:
        return np.unique(self.groundtruth_positions_train[:, -1])


_ADP_WORKER_CSI: np.ndarray | None = None


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def set_random_seeds(seed: int = DEFAULT_SEED, tf_module: Any | None = None) -> None:
    np.random.seed(seed)
    if tf_module is not None:
        tf_module.random.set_seed(seed)


def load_dataset_artifacts(
    *,
    include_floor_indices: bool = False,
    include_geometry: bool = False,
) -> DatasetArtifacts:
    csi_freq_domain_train = np.load("dataset/csi_freq_domain_train.npy")
    groundtruth_positions_train = np.load("dataset/groundtruth_positions_train.npy")
    groundtruth_floor_indices = (
        np.load("dataset/groundtruth_floor_indices.npy") if include_floor_indices else None
    )
    array_positions_3d = np.load("dataset/array_positions_3D.npy") if include_geometry else None
    center_positions_3d = np.load("dataset/center_positions_3D.npy") if include_geometry else None
    return DatasetArtifacts(
        csi_freq_domain_train=csi_freq_domain_train,
        groundtruth_positions_train=groundtruth_positions_train,
        groundtruth_floor_indices=groundtruth_floor_indices,
        array_positions_3d=array_positions_3d,
        center_positions_3d=center_positions_3d,
    )


def load_estimated_floor_indices(path: str | Path = "results/est_absolute_floor_indices.npy") -> np.ndarray:
    return np.load(path)


def make_floor_labels(floor_count: int, samples_per_floor: int) -> np.ndarray:
    return np.repeat(np.arange(floor_count, dtype=np.int32), int(samples_per_floor))


def floor_sample_indices(floor_indices: np.ndarray, floor_index: int) -> np.ndarray:
    return np.flatnonzero(np.asarray(floor_indices).reshape(-1) == floor_index)


def values_by_floor(values: np.ndarray, floor_indices: np.ndarray, floor_count: int) -> dict[int, np.ndarray]:
    return {
        floor_index: values[floor_sample_indices(floor_indices, floor_index)]
        for floor_index in range(floor_count)
    }


def reorder_by_floor(values: np.ndarray, floor_indices: np.ndarray, floor_count: int) -> np.ndarray:
    return np.concatenate(
        [values[floor_sample_indices(floor_indices, floor_index)] for floor_index in range(floor_count)],
        axis=0,
    )


def compute_array_normal_vectors(array_positions_3d: np.ndarray, center_positions_3d: np.ndarray) -> np.ndarray:
    normal_vectors = center_positions_3d[:, np.newaxis, :] - array_positions_3d
    return normal_vectors / np.linalg.norm(normal_vectors, axis=-1, keepdims=True)


def plot_scene_overview(
    groundtruth_positions: np.ndarray,
    array_positions_3d: np.ndarray,
    center_positions_3d: np.ndarray | None = None,
    title: str = "Randomly Generated 3D Positions in L-Shape",
) -> None:
    array_normalvectors_3d = None
    if center_positions_3d is not None:
        array_normalvectors_3d = compute_array_normal_vectors(array_positions_3d, center_positions_3d)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    for floor_index in range(array_positions_3d.shape[0]):
        ax.scatter(
            array_positions_3d[floor_index, :, 0],
            array_positions_3d[floor_index, :, 1],
            array_positions_3d[floor_index, :, 2],
            s=10,
            c="black",
            alpha=1,
        )
        if center_positions_3d is not None:
            ax.scatter(
                center_positions_3d[floor_index, 0],
                center_positions_3d[floor_index, 1],
                center_positions_3d[floor_index, 2],
                s=10,
                c="red",
                alpha=1,
            )
        if array_normalvectors_3d is not None:
            ax.scatter(
                array_positions_3d[floor_index, :, 0] + array_normalvectors_3d[floor_index, :, 0],
                array_positions_3d[floor_index, :, 1] + array_normalvectors_3d[floor_index, :, 1],
                array_positions_3d[floor_index, :, 2] + array_normalvectors_3d[floor_index, :, 2],
                s=10,
                c="green",
                alpha=1,
            )

    ax.scatter(
        groundtruth_positions[:, 0],
        groundtruth_positions[:, 1],
        groundtruth_positions[:, 2],
        s=1,
        c="blue",
        alpha=0.5,
    )
    ax.set_title(title)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.show()


def plot_floor_colored_positions(
    positions: np.ndarray,
    groundtruth_positions: np.ndarray,
    floor_indices: np.ndarray,
    ue_heights: np.ndarray,
    title: str | None = None,
    alpha: float = 1.0,
    save_path: str | Path | None = None,
) -> None:
    floor_indices = np.asarray(floor_indices, dtype=np.int32)
    fig = plt.figure(figsize=(7.5, 6), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        ue_heights[floor_indices],
        c=floor_indices,
        cmap="rainbow",
        s=5,
        alpha=alpha,
        linewidths=4,
    )

    ax.set_xlabel("x-coordinate $[m]$")
    ax.set_ylabel("y-coordinate $[m]$")
    ax.set_zlabel("")
    if title is not None:
        ax.set_title(title)
    ax.set_box_aspect([
        np.ptp(groundtruth_positions[:, 0]),
        np.ptp(groundtruth_positions[:, 1]),
        np.ptp(groundtruth_positions[:, 2]),
    ])
    ax.view_init(elev=20.0, azim=125, roll=0)
    fig.text(0.98, 0.52, "z-coordinate [m]", rotation=90, va="center", ha="right")
    if save_path is not None:
        fig.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.15)
    plt.show()


def plot_spatial_colorized_positions(
    positions: np.ndarray,
    groundtruth_positions: np.ndarray,
    title: str | None = None,
    alpha: float = 1.0,
    save_path: str | Path | None = None,
    array_positions_3d: np.ndarray | None = None,
) -> None:
    center_point = np.zeros(3, dtype=np.float32)
    center_point[0] = 0.5 * (
        np.min(groundtruth_positions[:, 0], axis=0) + np.max(groundtruth_positions[:, 0], axis=0)
    )
    center_point[1] = 0.5 * (
        np.min(groundtruth_positions[:, 1], axis=0) + np.max(groundtruth_positions[:, 1], axis=0)
    )
    center_point[2] = 0.5 * (
        np.min(groundtruth_positions[:, 2], axis=0) + np.max(groundtruth_positions[:, 2], axis=0)
    )
    normalize_data = lambda values: (values - np.min(values)) / (np.max(values) - np.min(values))
    rgb_values = np.zeros((groundtruth_positions.shape[0], 3))
    rgb_values[:, 0] = 1 - 0.9 * normalize_data(groundtruth_positions[:, 0])
    rgb_values[:, 1] = 0.8 * normalize_data(np.square(np.linalg.norm(groundtruth_positions - center_point, axis=1)))
    rgb_values[:, 2] = 0.9 * normalize_data(groundtruth_positions[:, 2])

    fig = plt.figure(figsize=(7.5, 6), constrained_layout=True)
    ax = fig.add_subplot(projection="3d")
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=rgb_values, s=5, alpha=alpha, linewidths=3)

    if array_positions_3d is not None:
        for floor_index in range(array_positions_3d.shape[0]):
            ax.scatter(
                array_positions_3d[floor_index, :, 0],
                array_positions_3d[floor_index, :, 1],
                array_positions_3d[floor_index, :, 2],
                marker="s",
                s=7,
                c="black",
                alpha=1,
                linewidth=5,
            )

    ax.set_xlabel("x-coordinate $[m]$")
    ax.set_ylabel("y-coordinate $[m]$")
    ax.set_zlabel("")
    if title is not None:
        ax.set_title(title)
    ax.view_init(elev=10.0, azim=125, roll=0)
    fig.text(0.98, 0.52, "z-coordinate [m]", rotation=90, va="center", ha="right")
    if save_path is not None:
        fig.savefig(save_path, dpi=600, bbox_inches="tight", pad_inches=0.15)
    plt.show()


def beamspace_features(csi: np.ndarray, pad_rows: int, pad_cols: int) -> tuple[np.ndarray, np.ndarray]:
    power_by_beam = []
    mean_delay_by_beam = []

    for datapoint_index in range(csi.shape[0]):
        datapoint_power = []
        datapoint_mean_delay = []

        for array_index in range(csi.shape[1]):
            csi_zeropadded = np.pad(
                np.einsum("brcs->bscr", csi[datapoint_index, array_index, ...]),
                [[0, 0], [0, 0], [pad_cols, pad_cols], [pad_rows, pad_rows]],
            )
            csi_zeropadded = np.fft.ifftshift(csi_zeropadded, axes=(-2, -1))

            beam_frequency_space = np.fft.fft2(csi_zeropadded)
            beam_frequency_space = np.fft.fftshift(beam_frequency_space, axes=(-2, -1))
            beam_frequency_space = np.einsum("bscr->bcrs", beam_frequency_space)

            datapoint_power.append(np.sum(np.abs(beam_frequency_space) ** 2, axis=-1))
            datapoint_mean_delay.append(
                np.angle(
                    np.sum(
                        beam_frequency_space[..., 1:] * np.conj(beam_frequency_space[..., :-1]),
                        axis=-1,
                    )
                )
            )

        power_by_beam.append(datapoint_power)
        mean_delay_by_beam.append(datapoint_mean_delay)

    return np.asarray(power_by_beam), np.asarray(mean_delay_by_beam)


def build_csi_features(
    csi_freq_domain: np.ndarray,
    elevation_res: int = 16,
    azimuth_res: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    real_rows = csi_freq_domain.shape[-3]
    real_cols = csi_freq_domain.shape[-2]
    pad_rows = elevation_res // 2 - real_rows // 2
    pad_cols = azimuth_res // 2 - real_cols // 2

    power_by_beam, mean_delay_by_beam = beamspace_features(csi_freq_domain, pad_rows, pad_cols)
    csi_features = np.concatenate(
        (power_by_beam[..., np.newaxis], mean_delay_by_beam[..., np.newaxis]),
        axis=-1,
    )
    csi_features = csi_features.reshape(
        (
            csi_features.shape[0],
            -1,
            csi_features.shape[3],
            csi_features.shape[4],
            csi_features.shape[5],
        )
    )
    return power_by_beam, mean_delay_by_beam, csi_features


def affine_transform_channel_chart(groundtruth_pos: np.ndarray, channel_chart_pos: np.ndarray) -> np.ndarray:
    pad = lambda values: np.hstack([values, np.ones((values.shape[0], 1))])
    unpad = lambda values: values[:, :-1]
    transform_matrix, _, _, _ = np.linalg.lstsq(
        pad(channel_chart_pos),
        pad(groundtruth_pos),
        rcond=None,
    )
    return unpad(np.dot(pad(channel_chart_pos), transform_matrix))


def plot_cdf(
    positions: np.ndarray,
    reference_positions: np.ndarray,
    title: str | None = None,
    x_limit: float = 4.0,
) -> None:
    errorvectors = reference_positions[:, :3] - positions
    errors = np.sqrt(np.sum(errorvectors[:, :3] ** 2, axis=1))
    count, bins_count = np.histogram(errors, bins=200)
    pdf = count / np.sum(count)
    cdf = np.cumsum(pdf)

    bins_count[0] = 0
    cdf = np.append([0], cdf)

    plt.figure(figsize=(5, 4))
    if title is not None:
        plt.title(title, fontsize=16)
    plt.plot(bins_count, cdf)
    plt.title("MAE = " + f"{np.mean(errors):.3f}m")
    plt.xlim((0, x_limit))
    plt.xlabel("Absolute Localization Error [m]")
    plt.ylabel("CDF")
    plt.grid()
    plt.show()


def plot_cdf_2d(
    positions: np.ndarray,
    reference_positions: np.ndarray,
    title: str | None = None,
    x_limit: float = 4.0,
) -> None:
    errorvectors = reference_positions[:, :2] - positions[:, :2]
    errors = np.sqrt(np.sum(errorvectors[:, :2] ** 2, axis=1))
    count, bins_count = np.histogram(errors, bins=200)
    pdf = count / np.sum(count)
    cdf = np.cumsum(pdf)

    bins_count[0] = 0
    cdf = np.append([0], cdf)

    plt.figure(figsize=(5, 4))
    if title is not None:
        plt.title(title, fontsize=16)
    plt.plot(bins_count, cdf)
    plt.title("MAE = " + f"{np.mean(errors):.3f}m")
    plt.xlim((0, x_limit))
    plt.xlabel("Absolute Localization Error [m]")
    plt.ylabel("CDF")
    plt.grid()
    plt.show()


def print_floor_classification_metrics(
    predicted_floor_indices: np.ndarray,
    true_floor_indices: np.ndarray,
) -> float:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    difference_count = np.sum(predicted_floor_indices != true_floor_indices)
    error_rate = 100 * difference_count / true_floor_indices.shape[0]
    print(f"Number of errors: {difference_count}")
    print(f"Error rate: {error_rate}%")
    print(f"Adjusted Rand Index (ARI): {adjusted_rand_score(predicted_floor_indices, true_floor_indices)}")
    print(f"Normalized Mutual Information (NMI): {normalized_mutual_info_score(predicted_floor_indices, true_floor_indices)}")
    return float(error_rate)


def _get_multiprocessing_context():
    import multiprocessing as mp

    if "fork" in mp.get_all_start_methods():
        return mp.get_context("fork")
    return None


def _compute_adp_row(csi_metric_input: np.ndarray, index: int) -> np.ndarray:
    h = csi_metric_input[index]
    w = csi_metric_input[index:]

    dotproducts = np.abs(
        np.einsum("brmt,lbrmt->lbt", np.conj(h), w, dtype=np.complex128)
    ) ** 2
    norms = np.real(
        np.einsum("brmt,brmt->bt", h, np.conj(h), dtype=np.complex128)
        * np.einsum("lbrmt,lbrmt->lbt", w, np.conj(w), dtype=np.complex128)
    )

    normalized_dotproducts = np.full(dotproducts.shape, np.nan, dtype=np.float64)
    np.divide(dotproducts, norms, out=normalized_dotproducts, where=norms != 0)
    return np.sum(np.float64(1.0) - normalized_dotproducts, axis=(1, 2), dtype=np.float64)


def _adp_dissimilarities_worker(todo_queue: Any, output_queue: Any) -> None:
    while True:
        index = todo_queue.get()
        if index == -1:
            output_queue.put((-1, None))
            break
        output_queue.put((index, _compute_adp_row(_ADP_WORKER_CSI, index)))


def replace_nan_with_scaled_max(values: np.ndarray, scale: float = 5.0) -> np.ndarray:
    filled_values = np.array(values, copy=True, dtype=np.float64)
    finite_values = filled_values[np.isfinite(filled_values)]
    if finite_values.size == 0:
        raise ValueError("Cannot replace non-finite values when all values are non-finite.")
    filled_values[~np.isfinite(filled_values)] = np.max(finite_values) * scale
    return filled_values


def compute_adp_dissimilarity_matrix(
    csi_metric_input: np.ndarray,
    process_count: int | None = None,
) -> np.ndarray:
    import multiprocessing as mp
    import tqdm

    sample_count = csi_metric_input.shape[0]
    adp_dissimilarity_matrix = np.zeros((sample_count, sample_count), dtype=np.float64)
    context = _get_multiprocessing_context()

    if context is None or sample_count < 2:
        with tqdm.tqdm(total=sample_count ** 2) as bar:
            for index in range(sample_count):
                row_values = _compute_adp_row(csi_metric_input, index)
                adp_dissimilarity_matrix[index, index:] = row_values
                adp_dissimilarity_matrix[index:, index] = row_values
                bar.update(2 * len(row_values) - 1)
        return replace_nan_with_scaled_max(adp_dissimilarity_matrix)

    global _ADP_WORKER_CSI
    _ADP_WORKER_CSI = csi_metric_input

    try:
        todo_queue = context.Queue()
        output_queue = context.Queue()
        processes = []
        worker_count = process_count or mp.cpu_count()

        for index in range(sample_count):
            todo_queue.put(index)

        for _ in range(worker_count):
            todo_queue.put(-1)
            process = context.Process(target=_adp_dissimilarities_worker, args=(todo_queue, output_queue))
            process.start()
            processes.append(process)

        finished_processes = 0
        with tqdm.tqdm(total=sample_count ** 2) as bar:
            while finished_processes != len(processes):
                index, row_values = output_queue.get()
                if index == -1:
                    finished_processes += 1
                    continue
                adp_dissimilarity_matrix[index, index:] = row_values
                adp_dissimilarity_matrix[index:, index] = row_values
                bar.update(2 * len(row_values) - 1)

        for process in processes:
            process.join()
    finally:
        _ADP_WORKER_CSI = None

    return replace_nan_with_scaled_max(adp_dissimilarity_matrix)


def compute_geodesic_dissimilarity_matrix(
    adp_dissimilarity_matrix: np.ndarray,
    n_neighbors: int = 20,
) -> np.ndarray:
    import scipy.sparse.csgraph
    import sklearn.neighbors

    nbrs_alg = sklearn.neighbors.NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="precomputed",
        n_jobs=-1,
    )
    nbrs = nbrs_alg.fit(adp_dissimilarity_matrix)
    nbg = sklearn.neighbors.kneighbors_graph(
        nbrs,
        n_neighbors,
        metric="precomputed",
        mode="distance",
    )
    return scipy.sparse.csgraph.dijkstra(nbg, directed=False)


def scale_dissimilarity_matrix_to_meters(
    dissimilarity_matrix: np.ndarray,
    reference_positions: np.ndarray,
    reduction: int = 30,
) -> np.ndarray:
    import scipy.spatial

    reduction = max(1, int(reduction))
    reduced_positions = reference_positions[::reduction]
    reduced_dissimilarity_matrix = dissimilarity_matrix[::reduction, ::reduction]
    reference_distance_matrix = scipy.spatial.distance_matrix(reduced_positions, reduced_positions)

    dissimilarity_unit_meters = np.full_like(reduced_dissimilarity_matrix, np.nan)
    np.divide(
        reduced_dissimilarity_matrix,
        reference_distance_matrix,
        out=dissimilarity_unit_meters,
        where=reference_distance_matrix != 0,
    )

    finite_values = dissimilarity_unit_meters[np.isfinite(dissimilarity_unit_meters)]
    if finite_values.size == 0:
        return np.array(dissimilarity_matrix, copy=True)

    scaling_factor_meters = np.median(finite_values)
    return dissimilarity_matrix / scaling_factor_meters


def plot_dissimilarity_over_euclidean_distance(
    dissimilarity_matrix: np.ndarray,
    distance_matrix: np.ndarray,
    label: str | None = None,
    bin_count: int = 200,
) -> None:
    dissimilarities_flat = dissimilarity_matrix.flatten()
    distances_flat = distance_matrix.flatten()

    max_distance = np.max(distances_flat)
    bins = np.linspace(0, max_distance, bin_count)
    bin_indices = np.digitize(distances_flat, bins)

    bin_medians = np.zeros(len(bins) - 1)
    bin_25_perc = np.zeros(len(bins) - 1)
    bin_75_perc = np.zeros(len(bins) - 1)
    for index in range(1, len(bins)):
        bin_values = dissimilarities_flat[bin_indices == index]
        if len(bin_values) > 0:
            bin_25_perc[index - 1], bin_medians[index - 1], bin_75_perc[index - 1] = np.percentile(
                bin_values,
                [25, 50, 75],
            )

    plt.plot(bins[:-1], bin_medians, label=label)
    plt.fill_between(bins[:-1], bin_25_perc, bin_75_perc, alpha=0.5)


def make_random_pair_dataset(
    csi_features_tensor: Any,
    dissimilarity_matrix_tensor: Any,
    *,
    seed_a: int,
    seed_b: int,
) -> Any:
    import tensorflow as tf

    datapoint_count = tf.shape(csi_features_tensor, out_type=tf.int64)[0]

    random_integer_pairs_dataset = tf.data.Dataset.zip(
        (
            tf.data.Dataset.random(seed=seed_a),
            tf.data.Dataset.random(seed=seed_b),
        )
    )

    @tf.function
    def fill_pairs(rand_a: Any, rand_b: Any) -> tuple[tuple[Any, Any], Any]:
        rand_a = tf.cast(rand_a, tf.int64)
        rand_b = tf.cast(rand_b, tf.int64)
        index_a = tf.math.floormod(rand_a, datapoint_count)
        index_b = tf.math.floormod(rand_b, datapoint_count)
        features = (csi_features_tensor[index_a], csi_features_tensor[index_b])
        label = dissimilarity_matrix_tensor[index_a, index_b]
        return features, label

    return random_integer_pairs_dataset.map(fill_pairs, num_parallel_calls=tf.data.AUTOTUNE)


def build_charting_models(input_shape: tuple[int, ...], embedding_dim: int) -> tuple[Any, Any]:
    import tensorflow as tf

    cc_embmodel_input = tf.keras.Input(shape=input_shape, name="input")
    cc_embmodel_output = tf.keras.layers.Flatten()(cc_embmodel_input)
    for units in (1024, 512, 256, 128, 64):
        cc_embmodel_output = tf.keras.layers.Dense(units, activation="relu")(cc_embmodel_output)
        cc_embmodel_output = tf.keras.layers.BatchNormalization()(cc_embmodel_output)
    cc_embmodel_output = tf.keras.layers.Dense(embedding_dim, activation="linear")(cc_embmodel_output)
    cc_embmodel = tf.keras.Model(inputs=cc_embmodel_input, outputs=cc_embmodel_output, name="ForwardChartingFunction")

    input_a = tf.keras.layers.Input(shape=input_shape)
    input_b = tf.keras.layers.Input(shape=input_shape)
    embedding_a = cc_embmodel(input_a)
    embedding_b = cc_embmodel(input_b)
    output = tf.keras.layers.concatenate([embedding_a, embedding_b], axis=1)
    model = tf.keras.models.Model([input_a, input_b], output, name="SiameseNeuralNetwork")
    return cc_embmodel, model


def make_distance_loss(embedding_dim: int):
    import tensorflow as tf

    def siamese_loss(y_true: Any, y_pred: Any) -> Any:
        pos_a, pos_b = y_pred[:, :embedding_dim], y_pred[:, embedding_dim:]
        distances_pred = tf.math.sqrt(tf.math.reduce_sum(tf.square(pos_a - pos_b), axis=1))
        return tf.reduce_mean(tf.square(distances_pred - tf.cast(y_true, tf.float32)))

    return siamese_loss


def train_siamese_model(
    model: Any,
    csi_features_tensor: Any,
    dissimilarity_matrix_tensor: Any,
    *,
    seed_a: int,
    seed_b: int,
    loss_fn: Any,
    training_schedule: tuple[tuple[float, int], ...] = DEFAULT_TRAINING_SCHEDULE,
    samples_per_session: int = 200000,
) -> None:
    import tensorflow as tf

    optimizer = tf.keras.optimizers.Adam()
    for session, (learning_rate, batch_size) in enumerate(training_schedule, start=1):
        session_seed_offset = 2 * (session - 1)
        random_pair_dataset = make_random_pair_dataset(
            csi_features_tensor,
            dissimilarity_matrix_tensor,
            seed_a=seed_a + session_seed_offset,
            seed_b=seed_b + session_seed_offset,
        )
        print("\nTraining Session ", session, "\nBatch Size: ", batch_size, "\nLearning rate: ", learning_rate)
        model.compile(loss=loss_fn, optimizer=optimizer)
        optimizer.learning_rate.assign(learning_rate)
        steps_per_epoch = samples_per_session // batch_size
        model.fit(
            random_pair_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE),
            steps_per_epoch=steps_per_epoch,
        )


def fit_channel_chart(
    csi_features: np.ndarray,
    dissimilarity_matrix: np.ndarray,
    *,
    embedding_dim: int,
    seed_a: int,
    seed_b: int,
    training_schedule: tuple[tuple[float, int], ...] = DEFAULT_TRAINING_SCHEDULE,
    samples_per_session: int = 200000,
) -> np.ndarray:
    import tensorflow as tf

    csi_features_tensor = tf.convert_to_tensor(csi_features)
    dissimilarity_matrix_tensor = tf.convert_to_tensor(dissimilarity_matrix, dtype=tf.float32)
    embedding_model, siamese_model = build_charting_models(csi_features.shape[1:], embedding_dim)
    train_siamese_model(
        siamese_model,
        csi_features_tensor,
        dissimilarity_matrix_tensor,
        seed_a=seed_a,
        seed_b=seed_b,
        loss_fn=make_distance_loss(embedding_dim),
        training_schedule=training_schedule,
        samples_per_session=samples_per_session,
    )
    return np.asarray(embedding_model.predict(csi_features_tensor))


def load_npz_array_dict(path: str | Path) -> dict[int, np.ndarray]:
    with np.load(path) as payload:
        return {
            int(key.removeprefix("floor_")): np.asarray(payload[key])
            for key in payload.files
        }


def save_npz_array_dict(path: str | Path, arrays_by_key: dict[int, np.ndarray]) -> None:
    np.savez(
        path,
        **{f"floor_{int(key)}": np.asarray(value) for key, value in arrays_by_key.items()},
    )