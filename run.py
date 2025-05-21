from flask import Flask
from flask_cors import CORS
from config import Config
from api.routes.dataset import dataset_bp
from api.routes.training import training_bp
from api.routes.inference import inference_bp
from api.routes.model import model_bp


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    CORS(app)

    app.register_blueprint(dataset_bp, url_prefix="/api/v1/datasets")
    app.register_blueprint(training_bp, url_prefix="/api/v1/training")
    app.register_blueprint(inference_bp, url_prefix="/api/v1/predict")
    app.register_blueprint(model_bp, url_prefix="/api/v1/models")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000)
