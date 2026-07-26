# Configuration settings for the ML backend project

import os

class Config:
    """Base configuration class."""
    DEBUG = False
    TESTING = False
    LOGGING_LEVEL = 'INFO'
    MODEL1_PATH = os.path.join(os.path.dirname(__file__), '../../models/model1_bilstm.pkl')
    MODEL2_PATH = os.path.join(os.path.dirname(__file__), '../../models/model2_cnn.pkl')

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOGGING_LEVEL = 'DEBUG'

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    LOGGING_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration."""
    LOGGING_LEVEL = 'ERROR'