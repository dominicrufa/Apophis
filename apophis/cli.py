"""Console script for apophis."""
import sys
import click


@click.command()
def main(args=None):
    """Console script for apophis."""
    click.echo("Replace this message by putting your code into "
               "apophis.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
