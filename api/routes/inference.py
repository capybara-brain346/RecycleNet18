from flask import Blueprint, request, jsonify
from api.services.inference_service import InferenceService

inference_bp = Blueprint("inference", __name__)
inference_service = InferenceService()


@inference_bp.route("", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        result = inference_service.predict(file)
        if not result:
            return jsonify({"error": "No production model available"}), 404
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@inference_bp.route("/logs", methods=["GET"])
def get_inference_logs():
    try:
        logs = inference_service.get_inference_logs()
        return jsonify(logs), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
