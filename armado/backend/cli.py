import pathlib
import shutil
import time

import click
import numpy as np

import gigapixel.acquire
import gigapixel.interfaces
import gigapixel.tools
from gigapixel import FpmConfig, FpmAcquisitionSetup, acquire_sequence, FpmStatus

ROI = ((0, 0), (128, 128))
PORT = "/dev/ttyACM0"

setup = FpmAcquisitionSetup.make_default()


@click.command()
@click.option(
    '--name',
    type=str,
    prompt="Sample name",
    default=setup.name,
    help="Sample name",
)
@click.option(
    '--description',
    type=str,
    prompt="Description",
    help="Description of sample",
)
@click.option(
    '--sample-dir',
    type=click.Path(
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    default=setup.sample_dir,
    help="Directory of samples"
)
@click.option(
    '--size',
    type=click.IntRange(0, 32),
    prompt="Illumination area",
    default=setup.size
)
@click.option(
    '--scheme',
    type=click.Choice(['square', 'circle', 'diamond'], case_sensitive=False),
    prompt="Scheme of illumination",
    default=setup.scheme,
    help="Scheme of illumination"
)
@click.option(
    '--colors',
    type=str,
    prompt="Colors",
    default=setup.colors,
    help="Colors used in acquisition",
)
@click.option(
    '--red-exp',
    type=click.IntRange(50, 5000000),
    default=setup.red_exp,
    help="Red central exposure value"
)
@click.option(
    '--green-exp',
    type=click.IntRange(50, 5000000),
    default=setup.green_exp,
    help="Green central exposure value"
)
@click.option(
    '--blue-exp',
    type=click.IntRange(50, 5000000),
    default=setup.blue_exp,
    help="Blue central exposure value"
)
@click.option(
    '--min-exposure',
    type=click.IntRange(50, 5000000),
    default=setup.min_exposure
)
@click.option(
    '--max-exposure',
    type=click.IntRange(50, 1000000),
    default=setup.max_exposure
)
@click.option(
    'config',
    '--fpm_config_file',
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    default=gigapixel.default_config_path(),
    help="Config file of fpm settings (.json)"
)
@click.option(
    'exposure_file',
    '--exposure_matrix',
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        path_type=pathlib.Path,
    ),
    default=None,
    help="Numpy file with exposure matrix (.npy)"
)
@click.option(
    '--bit-depth',
    type=click.IntRange(8, 16),
    default=setup.bit_depth,
    help="Camera images bit-depth"
)
@click.option(
    '--dry-run',
    is_flag=True,
    help="Simulate acquisition without save"
)
@click.option(
    '--dummy-devices',
    is_flag=True,
    help="Simulate acquisition with dummy devices"
)
@click.option(
    '--auto-start',
    is_flag=True,
    help="Start acquisition without confirmation prompt"
)
def acquire(
        name: str,
        description: str,
        sample_dir: pathlib.Path,
        size: int,
        scheme: str,
        colors: str,
        red_exp: int,
        green_exp: int,
        blue_exp: int,
        min_exposure: int,
        max_exposure: int,
        config: pathlib.Path,
        exposure_file: pathlib.Path,
        bit_depth: int,
        dry_run: bool,
        dummy_devices: bool,
        auto_start: bool,
):
    setup.name = name
    setup.description = description
    setup.sample_dir = sample_dir
    setup.size = size
    setup.scheme = scheme
    setup.colors = colors
    setup.red_exp = red_exp
    setup.green_exp = green_exp
    setup.blue_exp = blue_exp
    setup.min_exposure = min_exposure
    setup.max_exposure = max_exposure
    setup.bit_depth = bit_depth
    setup.dry_run = dry_run
    setup.dummy_devices = dummy_devices

    with open(config) as f:
        setup.config = FpmConfig.from_json(f.read())
    if exposure_file is None:
        setup.exposure_matrix = gigapixel.interfaces.default_exposure(
            setup.config.center
        )
    else:
        setup.exposure_matrix = np.load(exposure_file)

    eta = time.strftime('%H:%M:%S', time.gmtime(setup.get_estimated_time()))
    required_space = setup.get_estimated_size()
    free_space = shutil.disk_usage(setup.sample_dir).free / 1e9

    if dry_run:
        click.echo("DRY RUN (skip saving files in disk)")

    if dummy_devices:
        click.echo("DUMMY DEVICES (use fake camera and illumination driver)")

    click.echo(
        f"\nAcquisition require:\n"
        f"  Time: {eta}\n"
        f"  Space: {required_space:.2f}Gb / {free_space:.2f}Gb\n"
    )

    if required_space > free_space:
        click.echo("Insufficient space in disk")
        if not dry_run:
            raise click.Abort()

    if not auto_start:
        if not click.confirm("Continue?"):
            raise click.Abort()

    sequence_iterator = acquire_sequence(
        camera=gigapixel.set_camera(
            dummy=setup.dummy_devices,
            bit_depth=setup.bit_depth,
            roi=ROI
        ),
        illumination=gigapixel.set_illumination(
            dummy=setup.dummy_devices,
            port=PORT
        ),
        setup=setup,
        status=FpmStatus(setup) 
    )

    for image in sequence_iterator:
        if dry_run:
            print(image.get_name())
        else:
            current_name = image.save(sample_dir / name)
            print(current_name)


if __name__ == "__main__":
    acquire()
