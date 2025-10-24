import json
import numpy as np
from PIL import Image
import os

from .base_processor import ImageProcessor

from .pipeline_builder import PartialPipeline

def run_multi_pipeline(asset_dir: str, input_path: str, ProcessorClass: ImageProcessor, output_dir: str = '', debug: bool = False) -> Image.Image:
    config_path = os.path.join(asset_dir, "config.json")
    layout_path = os.path.join(asset_dir, 'layout.png')
    config = json.load(open(config_path))
    partials_json = config.get("partials", [])
    base = ProcessorClass.load(input_path)

    canvas, layout = base.load_layout(layout_path)
    partial_processors = []
    for partial_json in partials_json:
        print(f"Processing partial: {partial_json.get('id')}")
        pipeline = PartialPipeline.from_json(partial_json, asset_dir)
        proc = base.clone()
        for idx, step in enumerate(pipeline.steps):
            print(f"Processing step: {step.action_type}")
            step.execute(proc)
            if debug:
                proc.save(os.path.join(output_dir, f"debug_{partial_json.get('id')}_{idx}_{step.action_type}.png"), preview=False)
        partial_processors.append((proc, pipeline.location))

    for proc, loc in partial_processors:
        canvas.composite(proc, loc["left"], loc["top"])

    canvas.composite(layout, 0, 0)
    return canvas