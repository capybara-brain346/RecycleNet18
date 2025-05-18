from flask import Blueprint, request, jsonify
from api.services.model_service import ModelService

model_bp = Blueprint("model", __name__)
model_service = ModelService()


@model_bp.route("", methods=["GET"])
def list_models():
    try:
        models = model_service.list_models()
        return jsonify(models), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@model_bp.route("/<model_id>", methods=["GET"])
def get_model(model_id):
    try:
        model = model_service.get_model(model_id)
        if not model:
            return jsonify({"error": "Model not found"}), 404
        return jsonify(model), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@model_bp.route("/<model_id>/promote", methods=["POST"])
def promote_model(model_id):
    try:
        result = model_service.promote_model(model_id)
        if not result:
            return jsonify({"error": "Model not found"}), 404

        return jsonify(
            {"message": "Model promoted to production successfully", "model_id": result}
        ), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@model_bp.route("/production", methods=["GET"])
def get_production_model():
    try:
        model = model_service.get_production_model()
        if not model:
            return jsonify({"error": "No production model found"}), 404
        return jsonify(model), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
