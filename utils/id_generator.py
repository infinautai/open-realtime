from nanoid import generate
    
def generateId(prefix: str='id', len: int = 8) -> str:
    """
    Generate a random string of fixed length
    :param prefix: Prefix for the random string
    :param len: Length of the random string
    :return: Random string
    """

    # Generate a random string of the specified length

    alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return f'{prefix}_{generate(alphabet, len)}'
    
