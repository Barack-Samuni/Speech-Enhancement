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

    def process(self, noisy_signal, noise):
        """
        Processes the input signal using the wrapped filter.

        Args:
            input_signal: The signal to filter.

        Returns:
            The filtered signal.
        """
        return self.filter.process(noisy_signal, noise)
        