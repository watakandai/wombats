import subprocess as sp
import graphviz
import os
from IPython.display import Image, display


class FlexfringeInterface():
    """
    A nice python interface to the Flexfringe grammatical inference tool

    allows you to run grammatical inference on input files in abbadingo format
    and return the resulting dotfile data of the returned model
    """

    def __init__(self, binary_location: str='dfasat/flexfringe',
                 output_directory: str='./') -> 'FlexfringeInterface':
        """
        constructs an instance of the Flexfringe interface class instance

        :param      binary_location:   (absolute / relative) filepath to the
                                       flexfringe binary
        :type       binary_location:   str
        :param      output_directory:  The output directory of the learned
                                       model
        :type       output_directory:  str
        """

        self.binary_location = binary_location
        self._output_filename = 'dfa'
        self._output_base_filepath = None
        self._output_directory = output_directory

        self._flexfringe_output_dir_popt_str = 'o'

    def infer_model(self, training_file: str, **kwargs) -> {str, None}:
        """
        calls the flexfringe binary given the data in the training file

        :param      training_file:  The full filename path to the training data
        :type       training_file:  filepath str
        :param      kwargs:         keyword arguments to pass to flexfringe
                                    controlling the learning process
        :type       kwargs:         dictionary

        :returns:   string with the learned dot file or None if learning failed
        :rtype:     str or None
        """

        cmd = self._get_command(kwargs)
        output_file = self.learned_model_filepath

        flexfringe_call = [self.binary_location] + cmd + [training_file]

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
        :type       dot_file_data:  str
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
        :type       filepath:  str
        """
        (self._output_directory,
         self._output_base_filepath) = os.path.split(filepath)

    @property
    def learned_model_filepath(self) -> str:
        """
        the output filename for the fully learned model, as this is a
        different from the inputted "output-dir"

        :returns:   The learned model filepath.
        :rtype:     str
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
        :type       filepath:  str
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
        :type       kwargs:  dict

        :returns:   The list of commands.
        :rtype:     list of strings
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
