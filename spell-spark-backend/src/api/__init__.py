# Contents of /ml-backend-project/ml-backend-project/src/api/__init__.py

from .endpoints import router

def init_api(app):
    app.include_router(router)