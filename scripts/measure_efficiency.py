import argparse
import pickle
from pathlib import Path

import pandas
import torch
from torch.utils import benchmark

from segmentation import models
from segmentation.datasets import (
    OpenTextileSegmentation,
    SpectralWasteSegmentation,
)


DATASET_DEFAULT_PATHS = {
    "opentextile": Path("data/opentextile"),
    "spectralwaste": Path("data/spectralwaste_segmentation"),
}

components = None


def unimodal(model, input):
    with torch.inference_mode():
        model(input)


def unimodal_reduction(model, input):
    with torch.inference_mode():
        reduced_input = torch.nn.functional.conv2d(input, components)
        model(reduced_input)


def multimodal(model, input):
    with torch.inference_mode():
        model(input)


def multimodal_reduction(model, input):
    with torch.inference_mode():
        reduced_input = torch.nn.functional.conv2d(input[1], components)
        model([input[0], reduced_input])


def resolve_data_path(dataset: str, data_path: str):
    if data_path:
        return Path(data_path)
    return DATASET_DEFAULT_PATHS[dataset]


def load_num_classes(dataset: str, data_path: Path):
    if dataset == "spectralwaste":
        return len(SpectralWasteSegmentation.classes)
    if dataset == "opentextile":
        return OpenTextileSegmentation(
            str(data_path),
            split="train",
            input_mode="rgb",
            target_mode="labels",
            target_type="",
        ).num_classes
    raise ValueError(f"Unknown dataset: {dataset}")


def load_components(data_path: Path, device: str):
    reduction_path = data_path / "hyper_pca32" / "reduction_model.pkl"
    pca_model = pickle.load(open(reduction_path, "rb"))
    return torch.tensor(pca_model.components_)[..., None, None].to(
        dtype=torch.float32,
        device=device,
    )


def build_configs(num_classes: int, hyper_channels: int, reduced_hyper_channels: int):
    return {
        "mininet_rgb": (3, models.create_model("mininet", 3, num_classes), unimodal),
        # "mininet_hyper": (
        #     hyper_channels,
        #     models.create_model("mininet", hyper_channels, num_classes),
        #     unimodal,
        # ),
        # "mininet_hyper32": (
        #     hyper_channels,
        #     models.create_model("mininet", reduced_hyper_channels, num_classes),
        #     unimodal_reduction,
        # ),
        # "mininet_rgb_hyper": (
        #     [3, hyper_channels],
        #     models.create_model("mininet_multimodal", [3, hyper_channels], num_classes),
        #     multimodal,
        # ),
        "mininet_rgb_hyper32": (
            [3, hyper_channels],
            models.create_model(
                "mininet_multimodal", [3, reduced_hyper_channels], num_classes
            ),
            multimodal_reduction,
        ),
        "segformer_rgb": (
            3,
            models.create_model("segformer_b0", 3, num_classes),
            unimodal,
        ),
        # "segformer_hyper": (
        #     hyper_channels,
        #     models.create_model("segformer_b0", hyper_channels, num_classes),
        #     unimodal,
        # ),
        # "segformer_hyper32": (
        #     hyper_channels,
        #     models.create_model("segformer_b0", reduced_hyper_channels, num_classes),
        #     unimodal_reduction,
        # ),
        # "segformer_rgb_hyper": (
        #     [3, hyper_channels],
        #     models.create_model(
        #         "segformer_b0_multimodal", [3, hyper_channels], num_classes
        #     ),
        #     multimodal,
        # ),
        "segformer_rgb_hyper32": (
            [3, hyper_channels],
            models.create_model(
                "segformer_b0_multimodal", [3, reduced_hyper_channels], num_classes
            ),
            multimodal_reduction,
        ),
        "unet_rgb": (3, models.create_model("unet", 3, num_classes), unimodal),
        # "unet_hyper": (
        #     hyper_channels,
        #     models.create_model("unet", hyper_channels, num_classes),
        #     unimodal,
        # ),
        # "unet_hyper32": (
        #     hyper_channels,
        #     models.create_model("unet", reduced_hyper_channels, num_classes),
        #     unimodal_reduction,
        # ),
        # "unet_rgb_hyper": (
        #     [3, hyper_channels],
        #     models.create_model("unet_multimodal", [3, hyper_channels], num_classes),
        #     multimodal,
        # ),
        "unet_rgb_hyper32": (
            [3, hyper_channels],
            models.create_model(
                "unet_multimodal", [3, reduced_hyper_channels], num_classes
            ),
            multimodal_reduction,
        ),
        # "hyfuseunet_rgb_hyper": (
        #     [3, hyper_channels],
        #     models.create_model("hyfuseunet", [3, hyper_channels], num_classes),
        #     multimodal,
        # ),
        "hyfuseunet_rgb_hyper32": (
            [3, hyper_channels],
            models.create_model("hyfuseunet", [3, reduced_hyper_channels], num_classes),
            multimodal_reduction,
        ),
        # "mmsformer_rgb_hyper": (
        #     [3, hyper_channels],
        #     models.create_model("mmsformer_b0", [3, hyper_channels], num_classes),
        #     multimodal,
        # ),
        "mmsformer_rgb_hyper32": (
            [3, hyper_channels],
            models.create_model(
                "mmsformer_b0", [3, reduced_hyper_channels], num_classes
            ),
            multimodal_reduction,
        ),
        # "cmx_rgb_hyper": (
        #     [3, hyper_channels],
        #     models.create_model("cmx_b0", [3, hyper_channels], num_classes),
        #     multimodal,
        # ),
        "cmx_rgb_hyper32": (
            [3, hyper_channels],
            models.create_model("cmx_b0", [3, reduced_hyper_channels], num_classes),
            multimodal_reduction,
        ),
    }


def format_results_table(results: dict[str, dict[str, float]]):
    table = pandas.DataFrame(results).T.reset_index(names="model")
    table["parameters_m"] = table["parameters"] / 1e6
    table["gflops"] = table["flops"] / 1e9
    table["latency_ms"] = table["time"] * 1e3
    table["fps"] = table["fps"]
    table = table[["model", "parameters_m", "gflops", "latency_ms", "fps"]]
    table = table.rename(
        columns={
            "model": "Model",
            "parameters_m": "Params (M)",
            "gflops": "GFLOPs",
            "latency_ms": "Latency (ms)",
            "fps": "FPS",
        }
    )
    return table.to_string(
        index=False,
        justify="left",
        formatters={
            "Params (M)": "{:.3f}".format,
            "GFLOPs": "{:.3f}".format,
            "Latency (ms)": "{:.3f}".format,
            "FPS": "{:.3f}".format,
        },
    )


def maybe_synchronize(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def main(args):
    global components

    data_path = resolve_data_path(args.dataset, args.data_path)
    components = load_components(data_path, args.device)
    num_classes = load_num_classes(args.dataset, data_path)
    hyper_channels = components.shape[1]
    reduced_hyper_channels = components.shape[0]
    configs = build_configs(num_classes, hyper_channels, reduced_hyper_channels)
    results = {}

    print("dataset", args.dataset)
    print("data_path", data_path)
    print("components", components.numel())

    for name, (input_channels, model, inference_fn) in configs.items():
        print(name)

        result = {}
        model.to(args.device)
        model.eval()

        if isinstance(input_channels, list):
            input = [
                torch.randn(
                    args.batch_size,
                    num_channels,
                    args.image_size,
                    args.image_size,
                    dtype=torch.float32,
                    device=args.device,
                )
                for num_channels in input_channels
            ]
        else:
            input = torch.randn(
                args.batch_size,
                input_channels,
                args.image_size,
                args.image_size,
                dtype=torch.float32,
                device=args.device,
            )

        result["parameters"] = sum(p.numel() for p in model.parameters())

        with torch.inference_mode():
            for _ in range(args.warmup_runs):
                inference_fn(model, input)
            maybe_synchronize(args.device)

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                with_flops=True,
            ) as profiler:
                inference_fn(model, input)
            maybe_synchronize(args.device)

            timer = benchmark.Timer(
                stmt="inference_fn(model, input)",
                num_threads=args.num_threads,
                globals={
                    "model": model,
                    "input": input,
                    "inference_fn": inference_fn,
                },
            )

        result["flops"] = sum(event.flops for event in profiler.events())
        result["time"] = timer.timeit(args.num_runs).median
        result["fps"] = args.batch_size / result["time"]
        results[name] = result

    print()
    print(format_results_table(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        choices=sorted(DATASET_DEFAULT_PATHS.keys()),
        default="spectralwaste",
    )
    parser.add_argument("--data-path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-runs", type=int, default=100)
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--num-threads", type=int, default=32)

    main(parser.parse_args())

