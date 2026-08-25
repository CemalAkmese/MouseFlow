try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
    

def config_tensorflow(log_level: Literal['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'] = 'ERROR', allow_growth: bool = True):
    try:
        import tensorflow as tf
    except ImportError:
        # No TensorFlow installed -- fine when running DLC's PyTorch engine
        # (deeplabcut>=3.0), which doesn't need this GPU-memory-growth
        # config at all. This function predates that engine, back when
        # DLC/DGP only ran on TF1.
        return

    # tf.ConfigProto/tf.Session/tf.logging are TF1 API, removed from TF2's
    # top-level namespace (only reachable via tf.compat.v1 there) -- pick
    # whichever namespace actually has them so this works whether the env
    # has genuine TF1 (older DLC2/DGP Windows setups) or TF2 installed.
    tf1 = tf.compat.v1 if hasattr(tf, "compat") else tf

    config = tf1.ConfigProto()
    config.gpu_options.allow_growth = allow_growth
    tf1.Session(config=config)

    level = getattr(tf1.logging, log_level)  # gets a variable from a string from a module (in this case, using "ERROR" would get you tf.logging.ERROR)
    tf1.logging.set_verbosity(level)
