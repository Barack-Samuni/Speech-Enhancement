class FilterWrapper:
    """
    A wrapper to provide a unified API for different filter classes.
    Each filter class must implement `process(input_signal)` or similar.
    """

    def __init__(self, filter_cls, *args, **kwargs):
        """
        Args:
            filter_cls: The filter class to wrap.
            *args, **kwargs: Arguments to initialize the filter class.
        """
        self.filter = filter_cls(*args, **kwargs)

    def process(self, input_signal):
        """
        Processes the input signal using the wrapped filter.

        Args:
            input_signal: The signal to filter.

        Returns:
            The filtered signal.
        """
        # Try common method names
        if hasattr(self.filter, 'process'):
            return self.filter.process(input_signal)
        elif hasattr(self.filter, 'filter'):
            return self.filter.filter(input_signal)
        elif hasattr(self.filter, '__call__'):
            return self.filter(input_signal)
        else:
            raise NotImplementedError("The wrapped filter does not have a recognized process method.")