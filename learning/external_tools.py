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
                 output_directory: str = './'):
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
        self.output_filepath: str
        self.learned_model_filepath: str
        self.initial_model_filepath: str

        self._output_filename = 'dfa'
        self._final_output_addon_name = 'final'
        self._learned_model_filepath: str
        self._initial_output_addon_name = 'init_dfa'
        self._initial_model_filepath: str
        self._output_base_filepath = None
        self._output_directory = output_directory
        self._flexfringe_output_dir_popt_str = 'o'

    def infer_model(self, training_file: str = None,
                    get_help: bool = False,
                    go_fast: bool = False, **kwargs) -> {str, None}:
        """
        calls the flexfringe binary given the data in the training file

        :param      training_file:  The full filename path to the training data
        :param      get_help:       Whether or not to print the flexfringe
                                    usage help memu
        :param      go_fast:        optimizes this call to make it as fast
                                    as possible, at the expensive of usability.
                                    use for benchmarking / Hyperparam
                                    optimization.
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

            if not go_fast:
                # get summary statistics of learning data and save them for
                # later use of the inference interface
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

        if go_fast:
            stdout = sp.DEVNULL
        else:
            stdout = sp.PIPE

        completed_process = sp.run(flexfringe_call,
                                   stdout=stdout, stderr=sp.PIPE)
        if not go_fast:
            call_string = completed_process.stdout.decode()
            print('%s' % call_string)

        if not go_fast:
            model_data = self._read_model_data(output_file)
            if model_data is not None:
                return model_data
            else:
                print('No model output generated')
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

    def draw_initial_model(self) -> None:
        """
        Draws the initial (prefix-tree) model
        """

        dot_file = self.initial_model_filepath
        self.draw_IPython(self._read_model_data(dot_file))

    def draw_learned_model(self) -> None:
        """
        Draws the final, learned model
        """

        dot_file = self.learned_model_filepath
        self.draw_IPython(self._read_model_data(dot_file))

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

        addon_name = self._final_output_addon_name
        self._learned_model_filepath = self._get_model_file(addon_name)

        return self._learned_model_filepath

    @learned_model_filepath.setter
    def learned_model_filepath(self, filepath: str) -> None:
        """
        sets the learned_model_filepath and the base model's filepath

        :param      filepath:  The new learned model filepath.
        """

        addon_name = self._final_output_addon_name
        base_model_filepath = self._strip_model_file(filepath, addon_name)

        self._learned_model_filepath = filepath
        self.output_filepath = base_model_filepath

    @property
    def initial_model_filepath(self) -> str:
        """
        the output filename for the unlearned, initial model, as this is a
        different from the inputted "output-dir".

        In this case, it will be a prefix tree from the given learning data.

        :returns:   The initial model filepath.
        """

        addon_name = self._initial_output_addon_name
        self._initial_model_filepath = self._get_model_file(addon_name)

        return self._initial_model_filepath

    @initial_model_filepath.setter
    def initial_model_filepath(self, filepath: str) -> None:
        """
        sets the initial_model_filepath and the base model's filepath

        :param      filepath:  The new initial model filepath.
        """

        addon_name = self._initial_model_filepath
        base_model_filepath = self._strip_model_file(filepath, addon_name)

        self._learned_model_filepath = filepath
        self.output_filepath = base_model_filepath

    def _get_model_file(self, addon_name: str) -> str:
        """
        Gets the full model filepath, with the model type given by addon_name.

        :param      addon_name:  The name to append to the base model name
                                 to access the certain model file

        :returns:   The full model filepath string.
        """

        filepath = self.output_filepath
        f_dir, _ = os.path.split(filepath)

        full_model_filename = self._get_model_filename(addon_name)
        full_model_filepath = os.path.join(f_dir, full_model_filename)

        return full_model_filepath

    def _strip_model_file(self, model_filepath: str, addon_name: str) -> str:
        """
        Strips the full model filepath of its addon_name to get the base model
        filepath

        :param      model_filepath:  The full model filepath
        :param      addon_name:      The name to strip from the full model file

        :returns:   The base model filepath string.
        """

        f_dir, full_fname = os.path.split(model_filepath)
        fname, ext = os.path.splitext(full_fname)

        # base filepath is just the basename, before the "addon" model type
        # is added to the base model name
        if fname.endswith(addon_name):
            fname = fname[:-len(addon_name)]

        base_model_filepath = os.path.join(f_dir, fname)

        return base_model_filepath

    def _get_model_filename(self, addon_name: str) -> str:
        """
        Gets the model filename, with the model type given by addon_name.

        :param      addon_name:  The name to append to the base model name
                                 to access the certain model file

        :returns:   The model filename string.
        """

        filepath = self.output_filepath
        f_dir, full_fname = os.path.split(filepath)
        fname, ext = os.path.splitext(full_fname)

        full_model_filename = fname + addon_name + '.dot'

        return full_model_filename

    def _read_model_data(self, model_file: str) -> str:
        """
        Reads in the model data as a string.

        :param      model_file:  The model filepath

        :returns:   The model data as a string
        """

        try:
            with open(model_file) as fh:
                return fh.read()

        except FileNotFoundError:
            print('No model file was found.')

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
