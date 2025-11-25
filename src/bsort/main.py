"""
Main Entry Point for BSORT CLI.
Powered by Typer.
"""

import typer
from typing_extensions import Annotated

from bsort.infer import run_inference
from bsort.train import train_model
from bsort.utils import load_config

# Initialize CLI App
app = typer.Typer(
    name="bsort",
    help="Bottle Cap Sorting CLI Tool for Ada Mata MLE Test.",
    add_completion=False,
)


@app.command()
def train(
    config: Annotated[
        str, typer.Option("--config", "-c", help="Path to settings.yaml")
    ] = "configs/settings.yaml",
):
    """
    Starts the model training pipeline and exports to OpenVINO FP16.
    """
    try:
        print(f"📖 Loading configuration from: {config}")
        cfg = load_config(config)

        # Execute training
        output_model = train_model(cfg)

        typer.secho("\n🎉 Pipeline Finished Successfully!", fg=typer.colors.GREEN)
        typer.secho(f"Model ready at: {output_model}", fg=typer.colors.BLUE)

    except Exception as err:
        typer.secho(f"❌ Error: {err}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


@app.command()
def infer(
    config: Annotated[
        str, typer.Option("--config", "-c", help="Path to settings.yaml")
    ] = "configs/settings.yaml",
    image: Annotated[
        str, typer.Option("--image", "-i", help="Path to input image")
    ] = None,
):
    """
    Runs inference on a single image.
    """
    if image is None:
        typer.secho(
            "❌ Error: Please provide an image path using --image", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    try:
        cfg = load_config(config)
        run_inference(cfg, image)

    except Exception as err:
        typer.secho(f"❌ Inference Error: {err}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
