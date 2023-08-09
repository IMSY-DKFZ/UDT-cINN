import click
from subprocess import run

from src import settings

EXCLUDE_FILES = ['pai_images_qualitative_comparison.svg', 'semantic_reflectance_joint.svg']


def get_files_to_export():
    figures_dir = settings.figures_dir / 'manuscript'
    files = list(figures_dir.glob('*.svg'))
    return files


def get_command(f: str) -> list:
    cmd = ['inkscape', '--export-type', 'pdf', f]
    return cmd


def export_figures():
    files = get_files_to_export()
    for f in files:
        if f.name not in EXCLUDE_FILES:
            cmd = get_command(f=str(f))
            run(cmd)


@click.command()
@click.option('--export', is_flag=True, help="convert final figures to PDF for uploading to overleaf")
def main(export):
    if export:
        export_figures()


if __name__ == '__main__':
    main()
