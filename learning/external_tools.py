import subprocess as sp
import graphviz
import os
import re
from IPython.display import Image, display


class FlexfringeInterface():
    """
    A nice python interface to the Flexfringe grammatical inference tool

    allows you to run grammatical inference on input files in abbadingo format
    and return the resulting dotfile data of the returned model
    """

    def __init__(self, binary_location: str = 'dfasat/flexfringe',
                 output_directory: str = './') -> 'FlexfringeInterface':
        """
        constructs an instance of the Flexfringe interface class instance

        :param      binary_location:   (absolute / relative) filepath to the
                                       flexfringe binary
        :param      output_directory:  The output directory of the learned
                                       model
        """

        self.binary_location = binary_location
        self.num_training_examples: int
        self.num_symbols: int
        self.total_symbols_in_examples: int

        self._output_filename = 'dfa'
        self._output_base_filepath = None
        self._output_directory = output_directory
        self._flexfringe_output_dir_popt_str = 'o'

    def infer_model(self, training_file: str = None,
                    get_help: bool = False, **kwargs) -> {str, None}:
        """
        calls the flexfringe binary given the data in the training file

        :param      training_file:  The full filename path to the training data
        :param      kwargs:         keyword arguments to pass to flexfringe
                                    controlling the learning process

        :returns:   string with the learned dot file or None if learning failed
        """

        cmd = self._get_command(kwargs)
        output_file = self.learned_model_filepath

        if get_help:
            flexfringe_call = [self.binary_location] + cmd + ['']
        else:
            flexfringe_call = [self.binary_location] + cmd + [training_file]

            with open(training_file) as fh:
                content = fh.readlines()
                first_line = content[0]
                N, num_symbols_str = re.match(r'(\d*)\s(\d*)',
                                              first_line).groups()
                self.num_training_examples = int(N)
                self.num_symbols = int(num_symbols_str)

                self.total_symbols_in_examples = 0
                if self.num_training_examples > 0:
                    for line in content[1:]:
                        _, line_len, _ = re.match(r'(\d)\s(\d*)\s(.*)',
                                                  line).groups()
                        self.total_symbols_in_examples += int(line_len)

        if output_file is not None:
            try:
                os.remove(output_file)
            except OSError:
                pass

        callString = sp.run(flexfringe_call,
                            stdout=sp.PIPE, stderr=sp.PIPE).stdout.decode()
        print('%s' % callString)

        try:
            with open(output_file) as fh:
                return fh.read()

        except FileNotFoundError:
            print('No output file was generated.')

        return None

    def draw_IPython(self, dot_file_data: str) -> None:
        """
        Draws the dot file data in a way compatible with a jupyter / IPython
        notebook

        :param      dot_file_data:  The learned model dot file data
        """
        if dot_file_data == '':
            pass
        else:
            g = graphviz.Source(dot_file_data, format='png')
            g.render()
            display(Image(g.render()))

    @property
    def output_filepath(self) -> str:
        """The output filepath for the results of learning the model"""
        self._output_base_filepath = os.path.join(self._output_directory,
                                                  self._output_filename)

        return self._output_base_filepath

    @output_filepath.setter
    def output_filepath(self, filepath: str) -> None:
        """
        sets output_filepath and output_directory based on the given filepath

        :param      filepath:  The new filepath
        """
        (self._output_directory,
         self._output_base_filepath) = os.path.split(filepath)

    @property
    def learned_model_filepath(self) -> str:
        """
        the output filename for the fully learned model, as this is a
        different from the inputted "output-dir"

        :returns:   The learned model filepath.
        """

        filepath = self.output_filepath

        f_dir, full_fname = os.path.split(filepath)
        fname, ext = os.path.splitext(full_fname)
        full_learned_model_filename = fname + 'final' + '.dot'

        full_learned_model_filepath = os.path.join(f_dir,
                                                   full_learned_model_filename)
        self._learned_model_filepath = full_learned_model_filepath
        return self._learned_model_filepath

    @learned_model_filepath.setter
    def learned_model_filepath(self, filepath: str) -> None:
        """
        sets the learned_model_filepath

        :param      filepath:  The new learned model filepath.
        """

        f_dir, full_fname = os.path.split(filepath)
        fname, ext = os.path.splitext(full_fname)

        # output filepath is just the basename, before the 'final' model is
        # outputted
        if fname.endswith('final'):
            fname = fname[:-len('final')]

        self._learned_model_filepath = filepath
        self.output_filepath = os.path.join(f_dir, fname)

    def _get_command(self, kwargs: dict) -> list:
        """
        gets a list of popt commands to send the binary

        :param      kwargs:  The flexfringe tool keyword arguments

        :returns:   The list of commands.
        """

        # default argument is to print the program's man page
        if(len(kwargs) > 1):
            cmd = ['-' + key + '=' + kwargs[key] for key in kwargs]

            # need to give the output directory only if the user hasn't already
            # put that in kwargs.
            if self._flexfringe_output_dir_popt_str not in kwargs:
                cmd += ['--output-dir={}'.format(self.output_filepath)]
            else:
                key = self._flexfringe_output_dir_popt_str
                self.output_filepath = kwargs[key]
        else:
            cmd = ['--help']
            print('no learning options specified, printing tool help:')

        return cmd
