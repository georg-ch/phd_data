from pathlib import Path
import tomllib


def load_plot_config(
    section: str, *, pie_output: bool = False
) -> tuple[dict, dict, Path]:
    """Return plot params, global params, and the resolved HTML output path."""

    SRC_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SRC_DIR.parents[1]
    PARAMS_FILE = PROJECT_ROOT / "config" / "plotting_params.toml"

    with open(PARAMS_FILE, "rb") as f:
        config = tomllib.load(f)

    params = config[section]
    params_global = config["global"]

    out_path = (PROJECT_ROOT / params["out_folder"] / params["out_name"]).with_suffix(
        ".html"
    )
    if pie_output:
        out_path = out_path.with_name(f"{out_path.stem}_pie.html")

    return params, params_global, out_path
