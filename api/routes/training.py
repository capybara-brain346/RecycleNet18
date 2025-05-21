from flask import Blueprint, request, jsonify
from api.services.training_service import TrainingService

training_bp = Blueprint("training", __name__)
training_service = TrainingService()


@training_bp.route("/start", methods=["POST"])
def start_training():
    data = request.get_json()

    required_fields = ["dataset_id"]
    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    hyperparameters = data.get("hyperparameters", {})
    valid_hyperparams = {
        "epochs": int,
        "batch_size": int,
        "image_size": int,
        "patience": int,
        "device": str,
    }

    for param, value in hyperparameters.items():
        if param in valid_hyperparams:
            try:
                hyperparameters[param] = valid_hyperparams[param](value)
            except (ValueError, TypeError):
                return jsonify(
                    {"error": f"Invalid value for hyperparameter: {param}"}
                ), 400

    try:
        job_id = training_service.start_training(
            dataset_id=data["dataset_id"], hyperparameters=hyperparameters
        )
        return jsonify(
            {"message": "Training job started successfully", "job_id": job_id}
        ), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@training_bp.route("/jobs", methods=["GET"])
def list_jobs():
    try:
        jobs = training_service.list_jobs()
        return jsonify(jobs), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@training_bp.route("/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id):
    try:
        job = training_service.get_job_status(job_id)
        if not job:
            return jsonify({"error": "Job not found"}), 404
        return jsonify(job), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@training_bp.route("/jobs/<job_id>/metrics", methods=["GET"])
def get_training_metrics(job_id):
    try:
        metrics = training_service.get_training_metrics(job_id)
        return jsonify(metrics), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
