import numpy as np

def mock_get_input_price(input_reserve, output_reserve):
    input_amount_with_fee = 997
    numerator = input_amount_with_fee * output_reserve
    denominator = (input_reserve * 1000) + input_amount_with_fee
    return np.float128(numerator) / np.float128(denominator)


p = mock_get_input_price(1000000000000000000000, 1000000000000000000000)
