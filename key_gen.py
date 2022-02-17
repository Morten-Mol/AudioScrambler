"""Generation of encryption key for use in scrambling/de-scrambling."""
from numpy.random import default_rng
import hashlib


def _get_password():
    """Get clear text encryption password from user input.

    Args:
        None

    Returns:
        (string): Entered password
    """
    return input('Please enter password and press enter...\n')


def _create_generator(pw):
    """Create seeded Generator object for key generaation.

    Args:
        pw (string): Password entered by user

    Returns:
        generator (Numpy.random.generator): Seeded Generator object for rng
    """
    # Convert password to a bytestring and use it as input for SHA-512
    seed = hashlib.sha512(pw.encode())

    # Intiallize the Generator object of the pseudo random numbers, using the integer
    # version of the hashed password as the seed
    return default_rng(int(seed.hexdigest(), 16))


def _create_scrambling_key(Generator, n, w, max_freq):
    """Create scrambling key based on given Numpy Random Generator.

    Args:
        Generator(Numpy.random.generator): Random value generator seeded using password
        n (uint): Amount of shuffling operations to be done
        w (float): Bandwidth of the spectra to be shuffled, given in kHz
        max_freq(float): Maximum frequency limit of the shuffled spectra, given in kHz

    Returns:
        scrambling_key (list of dicts): List of dictionaries, each containing a set of frequencies to be shuffled.
    """
    # Initialize scrambling key
    scrambling_key = [{'b1_cfreq': None, 'b2_cfreq': None} for _ in range(n)]

    # Populatue all list entries in the key
    for i in range(n):
        # Select two pseudo-random center frequncies for shuffling
        band1_center_freq = Generator.uniform(low=w, high=max_freq)
        band2_center_freq = Generator.uniform(low=w, high=max_freq)

        # Check that the two selected bands are not too close to each other
        while abs(band1_center_freq-band2_center_freq) < w:
            band1_center_freq = Generator.uniform(low=w, high=max_freq)
            band2_center_freq = Generator.uniform(low=w, high=max_freq)

        # Put center frequencies values into key
        scrambling_key[i]['b1_cfreq'] = band1_center_freq
        scrambling_key[i]['b2_cfreq'] = band2_center_freq

    print('1st key item:' + str(scrambling_key[0]))
    print('last key' + str(scrambling_key[-1]))
    return scrambling_key


def _create_de_scrambling_key(Generator, n, w, max_freq):
    """Create de-scrambling key based on given Numpy Random Generator.

    Args:
        Generator(Numpy.random.generator): Random value generator seeded using password
        n (uint): Amount of shuffling operations to be done
        w (float): Bandwidth of the spectra to be shuffled, given in kHz
        max_freq(float): Maximum frequency limit of the shuffled spectra, given in kHz

    Returns:
        de_scrambling_key (list(dicts)): List of dictionaries, each containing a set of frequencies to be shuffled.
    """
    # Initialize de-scrambling key
    de_scrambling_key = [{'b1_cfreq': None, 'b2_cfreq': None} for _ in range(n)]

    # Populatue all list entries in the key in reversed order
    for i in reversed(range(n)):
        # Select two pseudo-random center frequncies for shuffling
        band1_center_freq = Generator.uniform(low=w, high=max_freq)
        band2_center_freq = Generator.uniform(low=w, high=max_freq)

        # Check that the two selected bands are not too close to each other
        while abs(band1_center_freq-band2_center_freq) < w:
            band1_center_freq = Generator.uniform(low=w, high=max_freq)
            band2_center_freq = Generator.uniform(low=w, high=max_freq)

        # Put center frequencies values into key
        de_scrambling_key[i]['b1_cfreq'] = band1_center_freq
        de_scrambling_key[i]['b2_cfreq'] = band2_center_freq

    print('1st key item:' + str(de_scrambling_key[0]))
    print('last key' + str(de_scrambling_key[-1]))
    return de_scrambling_key


def key_generator(type_of_scrambling, n, w, max_freq):
    """Generate key for scrambling/de-scrambling an audio sequence based on a given password.

    Args:
        type_of_scrambling (str): Flag to control if the action needed is 'scrambling' or 'de-scrambling'
        n (uint): Amount of shuffling operations to be done
        w (float): Bandwidth of the spectral bands to be shuffled, given in kHz
        max_freq (float): Max frequency to be shuffled, given in kHz

    Returns:
        (list(dict)): Key containing a list of dictionaries, each containing a set of frequencies to be shuffled
    """
    # Get password and transform it into seeded random number generator
    rng_generator = _create_generator(_get_password())

    # Create key for specific operation based on type of scrambling
    if type_of_scrambling == 'de-scrambling':
        return _create_de_scrambling_key(rng_generator, n, w, max_freq)
    elif type_of_scrambling == 'scrambling':
        # If there is not a "Scrambled Audio.wav" file, create scrambling instructions from it
        return _create_scrambling_key(rng_generator, n, w, max_freq)
    else:
        # We need a recording to start with, so we raise and error
        raise FileNotFoundError("No recordings found in current directory - Please add one")
