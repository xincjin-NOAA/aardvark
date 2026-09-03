"""
Observation configuration module - Python equivalent of observation_config.yaml

This module provides the same configuration data as observation_config.yaml
but in Python format for programmatic access.
"""

# Values to replace with NaN when loading data
FILL_VALUES = [99999997952.00, 1000000.00, 1000000000, 9999]

# Instrument weights
INSTRUMENT_WEIGHTS = {
    'diag_t': 1.0,
    'diag_q': 1.0,
    'diag_ps': 1.0,
    'diag_uv': 1.0,
    'ges': 1.0,
}

# Channel weights
CHANNEL_WEIGHTS = {
    'diag_t': [1.0] * 22,
    'diag_q': [1.0] * 15,
    'diag_ps': [1.0] * 3,
    'diag_uv': [1.0] * 24,
}

COMMON_RANGE = [150, 350]

from ocelot_vendor.obs_config_utils import apply_derived_instrument_dims  # noqa: E402

# Observation configuration
OBSERVATION_CONFIG = {
    "state": {
        "ges": {
            'source': 'zarr',
            'features': ['tmp_2maboveground', 'pres_surface', 'vgrd_10maboveground', 'ugrd_10maboveground', 'spfh_2maboveground',] ,
            # 'features': ['tmp_2maboveground', 'hgt_surface', 'pres_surface', 'vgrd_10maboveground',  'ugrd_10maboveground',
            #              'dpt_2maboveground',] ,
            "target_dim": 5,              # number of state variables
            "input_dim": 20,              # 2 * N + FEATURES_AUX_DIM + num_metadata + STATIC_CONTEXT_DIM,
            "anal_zarr_name": "anal_urma",   # parquet file base name for anal
            "zarr_name": "ges_urma",         # parquet file base name for ges
        }
    },
    'satellite': {
    },
    'conventional': {
        'diag_t': {
            'source': 'zarr',
            'zarr_name': 'surface_obs_t',
            'features': ['t'],
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
        },
        'diag_u': {
            'source': 'zarr',
            'zarr_name': 'surface_obs_u',
            'features': ['u'],
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
        },
        'diag_v': {
            'source': 'zarr',
            'zarr_name': 'surface_obs_v',
            'features': ['v'],
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
        },
        'diag_ps': {
            'source': 'zarr',
            'zarr_name': 'surface_obs_ps',
            'features': ['ps'],
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
        },
         'diag_q': {
            'source': 'zarr',
            'zarr_name': 'surface_obs_q',
            'features': ['q'],
            'encoder_hidden_layers': 2,
            'decoder_hidden_layers': 2,
        },
    }
}

apply_derived_instrument_dims(OBSERVATION_CONFIG)

# Feature statistics
FEATURE_STATS = {
    'diag_t': {
        't': [289.11, 11.93],
    },
     'diag_q': {
        'q': [0.01, 0.01],
    },

     'diag_ps': {
        'ps': [ 95853.9, 3856.44],
    },

    'diag_u': {
        'u': [-0.07, 2.54],
   
    },
    'diag_v': {
        'u': [-0.81, 4.06],
   
    },

    'ges': {
        'pres_surface': [95020.7, 4063.36],
        'vgrd_10maboveground': [0.82, 4.87],
        'ugrd_10maboveground': [0.05, 2.61], 
        'spfh_2maboveground': [0.01, 0.01],
        'tmp_2maboveground': [289, 11.95], 
    }
}

STATE_INCREMENT_STATS = {
    "tmp_2maboveground": [
        0.0034647618349895526,
        0.07001476349932656
    ],
    "pres_surface": [
        1.077697040534138,
        4.34615488592226
    ],
    "vgrd_10maboveground": [
        -0.00799981470978102,
        0.083828175611821
    ],
    "ugrd_10maboveground": [
        -0.006471472003871081,
        0.07079058308031667
    ],
    "spfh_2maboveground": [
        1.5229471766891314e-05,
        4.344852107939282e-05
    ]
}

