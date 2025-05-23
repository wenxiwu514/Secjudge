from abc import ABC, abstractmethod
import argparse


class API(ABC):
    def __init__(self, args=None):
        self.args = args

    @staticmethod
    def command_line_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--api_help',
            action='help')
        return parser

    @classmethod
    def from_command_line_args(cls, args):
        """
        Creating the API from command line arguments.

        Args:
            args: (List[str]):
            The command line arguments
        Returns:
            API:
                The API object.
        """
        args = cls.command_line_parser().parse_args(args)
        print(args)
        return cls(**vars(args), args=args)



    @abstractmethod
    def text_variation(self, samples, labels,
                       num_variations_per_sample, variation_type):

        pass
