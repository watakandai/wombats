# 3rd-party packages
import yaml


##
# @brief      Implements an abstract generic builder class to use with factory
#
class Builder:

    ##
    # @brief      Builder constructor. Just sets the internal instance
    #             reference to None
    #
    def __init__(self):

        self._instance = None
        self._configName = None

    ##
    # @brief      Abstract implementation of the constructor for the object to
    #             be built.
    #
    # @param      kwargs  The keywords arguments for the object to be built's
    #                     constructor
    #
    # @return     a concrete implentation of the object to be built
    #
    def __call__(self, **kwargs):

        return NotImplementedError

    ##
    # @brief      reads in the simulation parameters from a YAML config file
    #
    # @param      config_file_name  The YAML configuration file name
    #
    # @return     configuration data dictionary for the simulation
    #
    @staticmethod
    def load_YAML_config_data(config_file_name):

        with open(config_file_name, 'r') as stream:
            config_data = yaml.load(stream, Loader=yaml.Loader)

        return config_data
