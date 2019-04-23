from flask import Flask
from flask_bootstrap import Bootstrap

from poet_py.config import Config

app = Flask(__name__)
app.config.from_object(Config)
bootstrap = Bootstrap(app)

