"""Media related tools"""

import numpy as np


class CarryOverEffect():
    """Calculates the retention factor.

    Uses convolutions to calculate the retention factor, describing the 
    proportion of media pressure that carries over from one week to the next.

    Attributes
    ----------
    spending_series: array
        An array indicating weekly advertising spend
    strength: float
        A float indicating how much gets carried over
    lenght: int
        An integer indicating how long does it get carried over
    
    Methods
    -------
    get_padding_width_per_side()
        Calculates the number of zeros to add in the pad
    add_padding_to_array()
        Adds padding to the array to transform
    carry_over_pattern()
        Creates the carry over pattern to transform the spending series
    convolve()
        Applies transformation to the spending series

    """


    def __init__(self, spending_series: np.array, strength: float, lenght: int):
        """
        Parameters
        ----------
        spending_series: array
            An array indicating weekly advertising spend
        strength: float
            A float indicating how much gets carried over
        lenght: int
            An integer indicating how long does it get carried over
        """
        
        self.spending_series = spending_series
        self.strength = strength
        self.lenght = lenght


    def get_padding_width_per_side(self) -> int:
        """Calculates the number of zeros to add in the pad.

        Calculates the number of zeros to be added at the beginning and end of 
        the series (pad), to allow the application of convolution.

        Parameters
        ----------
        

        Raises
        ------
        int
            The number of zeros to be added in the next step
        """

        if self.lenght >= 1:
            padding_width = self.lenght - 1
        else:
            padding_width = 0
        return padding_width


    def add_padding_to_array(self) -> np.array:
        """Adds padding to the array to transform.

        Adds to the data series to be transformed, the zeros necessary to be 
        able to apply the convolution. Shape: (spending_series + padding_width).

        Parameters
        ----------
        

        Raises
        ------
        array
            An array with original values and padding
        """
        
        padding_width = self.get_padding_width_per_side()
        # Multiply with two because we need padding at the beginning and end.
        # Example, if spending_series.shape = (10, ) and padding = 2, then 
        # spending_series_with_padding.shape = (14, )
        spending_series_with_padding = np.zeros(
            self.spending_series.shape[0] + padding_width * 2)
        spending_series_with_padding[padding_width:-padding_width] = self.spending_series
        return spending_series_with_padding


    def carry_over_pattern(self) -> np.array:
        """Creates the carry over pattern to transform the spending series.

        It generates an array showing the decreasing evolution of the Retention 
        Factor over the weeks. The values that conform it are between (0, 1].

        Parameters
        ----------
        

        Raises
        ------
        array
            An array of values between (0, 1]
        """
        pattern = []
        for i in range(0, self.lenght):
            multiplier = self.strength ** (self.lenght - 1 - i)
            pattern.append(multiplier)
        return np.array(pattern)


    def convolve(self) -> np.array:
        """Applies transformation to the spending series.

        Uses the carry over pattern to transform the spending series by applying
        a convolution to simulate the Retention Factor over the weeks.

        Parameters
        ----------
        

        Raises
        ------
        array
            A transformed array
        """

        spending_series_with_padding = self.add_padding_to_array()
        spending_series_size = self.spending_series.shape[0]
        pattern = self.carry_over_pattern()
        pattern_size = pattern.shape[0]
        convolved_sepending_series = np.zeros(spending_series_size)
        for i in range(0, spending_series_size):            
            convolved_sepending_series[i] = np.sum(np.multiply(
                spending_series_with_padding[i: i + pattern_size], pattern))
        return convolved_sepending_series