# local packages
from wombats.factory.object_factory import ObjectFactory


class AutomatonCollection(ObjectFactory):
    """
    registering the builders for the different types of automaton objects
    with a more readable interface to our generic factory class.
    """

    def get(self, automaton_type, **config_data):
        """
        return an instance of an automaton given the automaton_type and the
        config_data.

        If the automaton has already been intialized with the same
        configuration data, it will return the already-initialized instance of
        it

        :param      automaton_type:  The automaton type
        :type       automaton_type:  string
        :param      config_data:     The keywords arguments to pass to the
                                     specific automaton builder class
        :type       config_data:     dictionary

        :returns:   an intialized / active reference to the desired type of
                    automaton object
        :rtype:     specific instance of an automaton object
        """

        return self.create(automaton_type, **config_data)
